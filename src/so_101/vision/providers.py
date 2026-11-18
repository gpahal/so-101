"""One adapter per provider. Each turns frames and a moment into a prediction dict:

    {"wrong": bool | None, "p_wrong": float | None, "category": str | None, "observation": str | None,
     "tokens_in": int | None, "tokens_out": int | None, "cost_usd": float | None, "latency_s": float, "raw": str}

`p_wrong` is a finite number in [0, 1] or None (`probability`), never NaN. A
chat model's answer whose `wrong` contradicts its `p_wrong` (wrong with p_wrong under 0.5, or not wrong
with it over 0.5) has `p_wrong` None and the number in `p_wrong_dropped`, so the supervisor acts on
neither side of it while failbench still scores `wrong`. A decision model's probability that isn't a
finite number is an error, not an answer.

It raises `ProviderError` (retryable or not) when there's no answer. `cost_usd` is None when the provider
reported no usage, so a caller keeping to a budget charges its own estimate instead of nothing. Each
request says which images it carries (`prompt.describe`) and labels each with `prompt.label`. SDKs are imported
here, lazily, so every other module and the tests run without the `failbench` dependency group.

- OpenRouter (Claude, GPT, Gemini): the `openai` SDK pointed at OpenRouter's OpenAI-compatible endpoint,
  with a strict JSON schema, `reasoning.effort`, and `provider.require_parameters` so a request is only
  routed to a host that honours the schema. OpenRouter returns each call's real cost in `usage.cost`.
- Gemini Robotics-ER 2: `google-genai` `generate_content` with a JSON schema and a thinking level.
- Gemini Decision 1 and OpenAI Decisions (`gpt-6-luna`): the providers' Decisions APIs, a yes/no question
  answered with a probability and a choice over the categories. `wrong` is the probability ≥ 0.5.
- Jev: `typesafe-sdk` `system_one`, with TypeSafe's opt-in image header.

Every adapter but Jev's has a request timeout of `TIMEOUT_S` (180 s) unless `timeout_s` says otherwise;
without one, Jev's keeps its SDK's own. `so101-failbench` waits for slow answers, while the supervisor passes
its `answer_within_s`, after which it stops waiting, so a call never runs on billing for an answer nothing reads. With a timeout given,
Jev's client also makes no retries. `adapter_for` keeps one adapter per model and timeout.

`openrouter_json` is the supervisor's escalation call, and `sdk_missing` says whether a provider's SDK
is installed.
"""

import base64
import json
import math
import os
import re
import threading
import time

from so_101.vision import prompt
from so_101.vision.models import (GEMINI, GEMINI_DECISIONS, OPENAI_DECISIONS, OPENROUTER, TYPESAFE, Model)

OPENROUTER_URL = "https://openrouter.ai/api/v1"
TIMEOUT_S = 180  # failbench's: it waits for a slow answer rather than score it as none


def _gemini_http(timeout_s: float) -> dict:
    """Preview models such as Robotics-ER 2 are only on v1beta (the SDK's default, v1, answers 404), and
    google-genai takes its timeout in whole milliseconds; without one a request can wait forever. Its
    client makes no retries unless asked."""
    return {"api_version": "v1beta", "timeout": round(timeout_s * 1000)}

_RETRYABLE = ("RateLimit", "Timeout", "Connection", "InternalServer", "Overloaded", "ServiceUnavailable", "ServerError")


class ProviderError(Exception):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode()


def _error(e: Exception) -> ProviderError:
    name = type(e).__name__
    status = getattr(e, "status_code", None) or getattr(e, "code", None) or getattr(e, "status", None)
    retry = any(r in name for r in _RETRYABLE) or status in (408, 409, 429, 500, 502, 503, 504, 529)
    return ProviderError(f"{name}: {e}"[:400], retryable=retry)


def _cost(m: Model, tokens_in: int | None, tokens_out: int | None) -> float | None:
    """The call's cost from its token counts at the model's prices, or None when there were none."""
    return None if tokens_in is None and tokens_out is None else m.cost(tokens_in, tokens_out)


def probability(value) -> float | None:
    """`value` as a probability: a finite number (or numeric string) clamped to [0, 1], else None. NaN and
    infinities are None, never clamped: `max(0.0, nan)` is 0.0, which would read as sure ok."""
    if value is None or isinstance(value, bool):
        return None
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, p)) if math.isfinite(p) else None


def _decided(base: dict, raw, answer: dict) -> dict:
    """A decision model's prediction from its raw probability `raw`: an error when it isn't a finite number."""
    p = probability(raw)
    if p is None:
        return {**base, "wrong": None, "error": f"no usable probability: {raw!r}", "raw": ""}
    return {**base, "wrong": p >= 0.5, "p_wrong": p, **answer, "raw": ""}


def parse_answer(text: str) -> dict:
    """The JSON answer from a chat model; tolerates code fences or text around the object."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", text or "", re.S)
        if not m:
            raise ProviderError(f"no JSON in the answer: {(text or '')[:200]!r}")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise ProviderError(f"unparseable answer: {e}: {(text or '')[:200]!r}")
    if not isinstance(data, dict):
        raise ProviderError(f"the answer isn't a JSON object: {(text or '')[:200]!r}")
    wrong = data.get("wrong")
    if isinstance(wrong, str):
        wrong = wrong.strip().lower() in ("true", "yes", "1")
    if not isinstance(wrong, bool):
        raise ProviderError(f"answer has no boolean 'wrong': {str(data)[:200]}")
    p = probability(data.get("p_wrong"))
    category = data.get("category")
    answer = {"wrong": wrong, "p_wrong": p, "category": category if category in prompt.CATEGORIES else "other",
              "observation": str(data.get("observation", ""))[:300]}
    if p is not None and (p < 0.5 if wrong else p > 0.5):  # it says one thing and its probability the other
        answer["p_wrong"], answer["p_wrong_dropped"] = None, p
    return answer


class OpenRouterChat:
    def __init__(self, m: Model, timeout_s: float | None = None):
        from openai import OpenAI
        self.m = m
        self.client = OpenAI(base_url=OPENROUTER_URL, api_key=os.environ["OPENROUTER_API_KEY"], timeout=timeout_s or TIMEOUT_S,
                             max_retries=0, default_headers={"X-Title": "so101-failbench"})

    def predict(self, images: list[tuple[str, bytes]], moment: str | None) -> dict:
        content = [{"type": "text", "text": prompt.describe(images)}]
        for cam, data in images:
            content.append({"type": "text", "text": f"{prompt.label(cam)}:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(data)}"}})
        content.append({"type": "text", "text": prompt.question(moment)})
        extra = {"usage": {"include": True}, "provider": {"require_parameters": True}}
        if self.m.effort:
            extra["reasoning"] = {"effort": self.m.effort}
        started = time.monotonic()
        try:
            r = self.client.chat.completions.create(
                model=self.m.model_id,
                messages=[{"role": "system", "content": prompt.SYSTEM}, {"role": "user", "content": content}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "verdict", "strict": True, "schema": prompt.ANSWER_SCHEMA}},
                max_tokens=24000 if self.m.effort in ("high", "xhigh", "max") else 8000,
                extra_body=extra,
            )
        except Exception as e:
            raise _error(e)
        latency = time.monotonic() - started
        if not r.choices:
            raise ProviderError(f"no choices in the response: {str(r)[:300]}", retryable=True)
        choice = r.choices[0]
        text = choice.message.content or ""
        refusal = getattr(choice.message, "refusal", None)
        u = r.usage
        tokens_in = getattr(u, "prompt_tokens", None)
        tokens_out = getattr(u, "completion_tokens", None)
        cost = getattr(u, "cost", None)
        if cost is None and u is not None and getattr(u, "model_extra", None):
            cost = u.model_extra.get("cost")
        base = {"tokens_in": tokens_in, "tokens_out": tokens_out, "latency_s": round(latency, 3),
                "cost_usd": float(cost) if cost is not None else _cost(self.m, tokens_in, tokens_out),
                "finish_reason": choice.finish_reason, "served_by": getattr(r, "provider", None) or (r.model_extra or {}).get("provider")}
        if refusal or choice.finish_reason in ("content_filter", "refusal"):
            return {**base, "wrong": None, "error": f"refused: {refusal or choice.finish_reason}", "raw": text[:500]}
        try:
            answer = parse_answer(text)
        except ProviderError as e:
            return {**base, "wrong": None, "error": str(e), "raw": text[:500]}
        return {**base, **answer, "raw": text[:500]}


class GeminiChat:
    """Gemini Robotics-ER 2 through `google-genai`."""

    def __init__(self, m: Model, timeout_s: float | None = None):
        from google import genai
        self.m = m
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=_gemini_http(timeout_s or TIMEOUT_S))

    def predict(self, images, moment):
        from google.genai import types
        contents = [prompt.describe(images)]
        for cam, data in images:
            contents += [f"{prompt.label(cam)}:", types.Part.from_bytes(data=data, mime_type="image/jpeg")]
        contents.append(prompt.question(moment))
        config = types.GenerateContentConfig(
            system_instruction=prompt.SYSTEM, response_mime_type="application/json",
            response_json_schema=prompt.ANSWER_SCHEMA,
            thinking_config=types.ThinkingConfig(thinking_level=self.m.effort) if self.m.effort else None)
        started = time.monotonic()
        try:
            r = self.client.models.generate_content(model=self.m.model_id, contents=contents, config=config)
        except Exception as e:
            raise _error(e)
        latency = time.monotonic() - started
        um = r.usage_metadata
        tokens_in = getattr(um, "prompt_token_count", None)
        tokens_out = None if um is None else \
            (getattr(um, "candidates_token_count", None) or 0) + (getattr(um, "thoughts_token_count", None) or 0)
        base = {"tokens_in": tokens_in, "tokens_out": tokens_out, "latency_s": round(latency, 3),
                "cost_usd": _cost(self.m, tokens_in, tokens_out)}
        finish = str(r.candidates[0].finish_reason) if r.candidates else "none"
        if not r.candidates or getattr(r.prompt_feedback, "block_reason", None):
            return {**base, "wrong": None, "error": f"blocked: {r.prompt_feedback}", "raw": ""}
        text = r.text or ""
        try:
            answer = parse_answer(text)
        except ProviderError as e:
            return {**base, "wrong": None, "error": f"{e} (finish {finish})", "raw": text[:500]}
        return {**base, **answer, "raw": text[:500], "finish_reason": finish}


def _category_criteria() -> dict[str, str]:
    return dict(prompt.CATEGORIES)


class GeminiDecisions:
    def __init__(self, m: Model, timeout_s: float | None = None):
        from google import genai
        self.m = m
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=_gemini_http(timeout_s or TIMEOUT_S))

    def predict(self, images, moment):
        inputs = [{"type": "text", "text": prompt.decision_context(moment, images) + "\n"}]
        for cam, data in images:
            inputs.append({"type": "text", "text": f"{prompt.label(cam)}:"})
            inputs.append({"type": "image", "data": b64(data), "mime_type": "image/jpeg", "resolution": "high"})
        started = time.monotonic()
        try:
            r = self.client.decisions.create(model=self.m.model_id, input=inputs, questions={
                "wrong": {"type": "noul", "instructions": prompt.WRONG_QUESTION},
                "category": {"type": "choice", "instructions": prompt.CATEGORY_QUESTION, "criteria": _category_criteria()},
            })
        except Exception as e:
            raise _error(e)
        latency = time.monotonic() - started
        tokens_in = getattr(r.usage, "total_input_tokens", None) if getattr(r, "usage", None) else None
        base = {"tokens_in": tokens_in, "tokens_out": 0, "latency_s": round(latency, 3), "cost_usd": _cost(self.m, tokens_in, None)}
        cat = r.answers["category"]
        return _decided(base, r.answers["wrong"].noul,
                        {"category": cat.choice, "category_confidence": getattr(cat, "confidence", None)})


class OpenAIDecisions:
    def __init__(self, m: Model, timeout_s: float | None = None):
        from openai import OpenAI
        self.m = m
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=timeout_s or TIMEOUT_S, max_retries=0)

    def predict(self, images, moment):
        content = [{"type": "input_text", "text": prompt.decision_context(moment, images)}]
        for cam, data in images:
            content.append({"type": "input_text", "text": f"{prompt.label(cam)}:"})
            content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64(data)}", "detail": "high"})
        started = time.monotonic()
        try:
            r = self.client.decisions.create(model=self.m.model_id, input=[{"role": "user", "content": content}], questions=[
                {"type": "predicate", "name": "wrong", "instructions": prompt.WRONG_QUESTION},
                {"type": "choice", "name": "category", "instructions": prompt.CATEGORY_QUESTION,
                 "choices": [{"value": k, "description": v} for k, v in prompt.CATEGORIES.items()]},
            ])
        except Exception as e:
            raise _error(e)
        latency = time.monotonic() - started
        tokens_in = getattr(r.usage, "input_tokens", None) if getattr(r, "usage", None) else None
        base = {"tokens_in": tokens_in, "tokens_out": 0, "latency_s": round(latency, 3), "cost_usd": _cost(self.m, tokens_in, None)}
        p = category = None
        for a in r.answers:
            if a.type == "refusal":
                return {**base, "wrong": None, "error": f"refused: {getattr(a, 'refusal', a)}", "raw": ""}
            if a.type == "predicate":
                p = a.probability
            elif a.type == "choice":
                category = a.choice
        if p is None:
            return {**base, "wrong": None, "error": f"no predicate answer: {str(r.answers)[:200]}", "raw": ""}
        return _decided(base, p, {"category": category})


class TypeSafeJev:
    def __init__(self, m: Model, timeout_s: float | None = None):
        """Without `timeout_s`, the SDK's own: 10 s a request, two retries within 30 s. With it, one
        request of at most `timeout_s`: a retry couldn't answer within it."""
        from typesafe_sdk import RetryPolicy, TypeSafeClient
        self.m = m
        self.client = TypeSafeClient(model=m.model_id) if timeout_s is None else \
            TypeSafeClient(model=m.model_id, timeout=timeout_s, retry=RetryPolicy(max_retries=0))

    def predict(self, images, moment):
        from typesafe_sdk import Choice, Image, Noul
        state = {"context": prompt.decision_context(moment, images)}
        for cam, data in images:
            state[cam] = Image(data)
        refs = " and ".join(f"`{cam}`" for cam, _ in images)
        started = time.monotonic()
        try:
            r = self.client.system_one(state=state, questions={
                "wrong": Noul(instructions=f"Given `context` and the camera images {refs}: {prompt.WRONG_QUESTION}"),
                "category": Choice(instructions=f"Given `context` and the camera images {refs}: {prompt.CATEGORY_QUESTION}",
                                   criteria=_category_criteria()),
            }, extra_headers={"TypeSafe-Danger-Zone-Images": "true"})
        except Exception as e:
            raise _error(e)
        latency = time.monotonic() - started
        tokens_in = getattr(r.usage, "input_tokens", None) if getattr(r, "usage", None) else None
        base = {"tokens_in": tokens_in, "tokens_out": 0, "latency_s": round(latency, 3),
                "cost_usd": _cost(self.m, tokens_in, None), "served_by": getattr(r, "model", None)}
        return _decided(base, r.nouls["wrong"].noul, {"category": r.choices["category"].choice})


_ADAPTERS = {OPENROUTER: OpenRouterChat, GEMINI: GeminiChat, GEMINI_DECISIONS: GeminiDecisions,
             OPENAI_DECISIONS: OpenAIDecisions, TYPESAFE: TypeSafeJev}
_cache: dict[str, object] = {}  # model key -> adapter with its default timeout
_timed: dict[tuple[str, float], object] = {}  # (model key, timeout_s) -> adapter with that timeout
_lock = threading.Lock()


def adapter_for(m: Model, timeout_s: float | None = None):
    """`m`'s adapter with a request timeout of `timeout_s`, or its default without one, made once per
    timeout and kept. A failure to make it is a non-retryable `ProviderError`, and isn't kept: the next
    call tries again."""
    cache, key = (_cache, m.key) if timeout_s is None else (_timed, (m.key, timeout_s))
    with _lock:
        if key not in cache:
            try:
                cache[key] = _ADAPTERS[m.provider](m) if timeout_s is None else _ADAPTERS[m.provider](m, timeout_s)
            except ImportError as e:
                raise ProviderError(f"{e}. The provider SDKs are in the failbench group: "
                                    "`uv run --group failbench so101-failbench ...`") from e
            except Exception as e:  # a missing key (KeyError) or a client that refused its settings
                raise ProviderError(f"{m.key}'s client couldn't be made: {type(e).__name__}: {e}"[:400]) from e
        return cache[key]


def sdk_missing(provider: str) -> str | None:
    """Why `provider`'s SDK can't be imported, or None when it can."""
    module = {OPENROUTER: "openai", OPENAI_DECISIONS: "openai", GEMINI: "google.genai",
              GEMINI_DECISIONS: "google.genai", TYPESAFE: "typesafe_sdk"}[provider]
    try:
        __import__(module)
    except ImportError:
        return f"the {module.split('.')[0].replace('_', '-')} SDK isn't installed (uv sync --group failbench)"
    return None


def openrouter_json(m: Model, system: str, image: bytes, text: str, name: str, schema: dict, *,
                    timeout: float = 60, max_tokens: int = 4000, title: str = "so101") -> tuple[dict, float | None]:
    """One OpenRouter chat call with a JPEG and text, answered as JSON matching `schema`. Returns the
    answer and the call's cost in USD: OpenRouter's, or the model's prices times the tokens, or None when the
    reply says neither, so the caller charges its estimate rather than nothing. Raises on failure."""
    from openai import OpenAI

    client = OpenAI(base_url=OPENROUTER_URL, api_key=os.environ[m.env], timeout=timeout, max_retries=1,
                    default_headers={"X-Title": title})
    r = client.chat.completions.create(
        model=m.model_id,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": [
                      {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(image)}"}},
                      {"type": "text", "text": text}]}],
        response_format={"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
        max_tokens=max_tokens,
        extra_body={"usage": {"include": True}, "provider": {"require_parameters": True},
                    **({"reasoning": {"effort": m.effort}} if m.effort else {})},
    )
    usage = r.usage
    cost = getattr(usage, "cost", None) or (getattr(usage, "model_extra", None) or {}).get("cost")
    tokens_in, tokens_out = getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)
    if cost is not None:
        cost = float(cost)
    elif tokens_in is not None and tokens_out is not None:
        cost = m.cost(tokens_in, tokens_out)
    return json.loads(r.choices[0].message.content or "{}"), cost
