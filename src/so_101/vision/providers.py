"""One adapter per provider. Each turns frames and a moment into a prediction dict:

    {"wrong": bool | None, "p_wrong": float | None, "category": str | None, "observation": str | None,
     "tokens_in": int | None, "tokens_out": int | None, "cost_usd": float, "latency_s": float, "raw": str}

and raises `ProviderError` (retryable or not) when there's no answer. SDKs are imported here,
lazily, so every other module and the tests run without the `failbench` dependency group.

- OpenRouter (Claude, GPT, Gemini): the `openai` SDK pointed at OpenRouter's OpenAI-compatible endpoint,
  with a strict JSON schema, `reasoning.effort`, and `provider.require_parameters` so a request is only
  routed to a host that honours the schema. OpenRouter returns each call's real cost in `usage.cost`.
- Gemini Robotics-ER 2: `google-genai` `generate_content` with a JSON schema and a thinking level.
- Gemini Decision 1 and OpenAI Decisions (`gpt-6-luna`): the providers' Decisions APIs, a yes/no question
  answered with a probability and a choice over the categories. `wrong` is the probability ≥ 0.5.
- Jev: `typesafe-sdk` `system_one`, with TypeSafe's opt-in image header.

`openrouter_json` is the supervisor's escalation call, and `sdk_missing` says whether a provider's SDK
is installed.
"""

import base64
import json
import os
import re
import threading
import time

from so_101.vision import prompt
from so_101.vision.models import (GEMINI, GEMINI_DECISIONS, OPENAI_DECISIONS, OPENROUTER, TYPESAFE, Model)

OPENROUTER_URL = "https://openrouter.ai/api/v1"
TIMEOUT_S = 180
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
    wrong = data.get("wrong")
    if isinstance(wrong, str):
        wrong = wrong.strip().lower() in ("true", "yes", "1")
    if not isinstance(wrong, bool):
        raise ProviderError(f"answer has no boolean 'wrong': {str(data)[:200]}")
    p = data.get("p_wrong")
    try:
        p = min(1.0, max(0.0, float(p))) if p is not None else None
    except (TypeError, ValueError):
        p = None
    category = data.get("category")
    return {"wrong": wrong, "p_wrong": p, "category": category if category in prompt.CATEGORIES else "other",
            "observation": str(data.get("observation", ""))[:300]}


class OpenRouterChat:
    def __init__(self, m: Model):
        from openai import OpenAI
        self.m = m
        self.client = OpenAI(base_url=OPENROUTER_URL, api_key=os.environ["OPENROUTER_API_KEY"], timeout=TIMEOUT_S,
                             max_retries=0, default_headers={"X-Title": "so101-failbench"})

    def predict(self, images: list[tuple[str, bytes]], moment: str | None) -> dict:
        content = []
        for cam, data in images:
            content.append({"type": "text", "text": f"{cam.upper()} camera:"})
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
                "cost_usd": float(cost) if cost is not None else self.m.cost(tokens_in, tokens_out),
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

    def __init__(self, m: Model):
        from google import genai
        self.m = m
        # Preview models such as Robotics-ER 2 are only on v1beta; the SDK's default here was v1 (a 404).
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options={"api_version": "v1beta"})

    def predict(self, images, moment):
        from google.genai import types
        contents = []
        for cam, data in images:
            contents += [f"{cam.upper()} camera:", types.Part.from_bytes(data=data, mime_type="image/jpeg")]
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
        tokens_out = (getattr(um, "candidates_token_count", None) or 0) + (getattr(um, "thoughts_token_count", None) or 0)
        base = {"tokens_in": tokens_in, "tokens_out": tokens_out, "latency_s": round(latency, 3),
                "cost_usd": self.m.cost(tokens_in, tokens_out)}
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
    def __init__(self, m: Model):
        from google import genai
        self.m = m
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options={"api_version": "v1beta"})

    def predict(self, images, moment):
        inputs = [{"type": "text", "text": prompt.decision_context(moment) + "\n"}]
        for cam, data in images:
            inputs.append({"type": "text", "text": f"{cam.upper()} camera:"})
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
        p = float(r.answers["wrong"].noul)
        cat = r.answers["category"]
        return {"wrong": p >= 0.5, "p_wrong": p, "category": cat.choice, "category_confidence": getattr(cat, "confidence", None),
                "tokens_in": tokens_in, "tokens_out": 0, "latency_s": round(latency, 3),
                "cost_usd": self.m.cost(tokens_in, 0), "raw": ""}


class OpenAIDecisions:
    def __init__(self, m: Model):
        from openai import OpenAI
        self.m = m
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=TIMEOUT_S, max_retries=0)

    def predict(self, images, moment):
        content = [{"type": "input_text", "text": prompt.decision_context(moment)}]
        for cam, data in images:
            content.append({"type": "input_text", "text": f"{cam.upper()} camera:"})
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
        base = {"tokens_in": tokens_in, "tokens_out": 0, "latency_s": round(latency, 3), "cost_usd": self.m.cost(tokens_in, 0)}
        p = category = None
        for a in r.answers:
            if a.type == "refusal":
                return {**base, "wrong": None, "error": f"refused: {getattr(a, 'refusal', a)}", "raw": ""}
            if a.type == "predicate":
                p = float(a.probability)
            elif a.type == "choice":
                category = a.choice
        if p is None:
            return {**base, "wrong": None, "error": f"no predicate answer: {str(r.answers)[:200]}", "raw": ""}
        return {**base, "wrong": p >= 0.5, "p_wrong": p, "category": category, "raw": ""}


class TypeSafeJev:
    def __init__(self, m: Model):
        from typesafe_sdk import TypeSafeClient
        self.m = m
        self.client = TypeSafeClient(model=m.model_id)

    def predict(self, images, moment):
        from typesafe_sdk import Choice, Image, Noul
        state = {"context": prompt.decision_context(moment)}
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
        p = float(r.nouls["wrong"].noul)
        return {"wrong": p >= 0.5, "p_wrong": p, "category": r.choices["category"].choice, "tokens_in": tokens_in,
                "tokens_out": 0, "latency_s": round(latency, 3), "cost_usd": self.m.cost(tokens_in, 0),
                "served_by": getattr(r, "model", None), "raw": ""}


_ADAPTERS = {OPENROUTER: OpenRouterChat, GEMINI: GeminiChat, GEMINI_DECISIONS: GeminiDecisions,
             OPENAI_DECISIONS: OpenAIDecisions, TYPESAFE: TypeSafeJev}
_cache: dict[str, object] = {}
_lock = threading.Lock()


def adapter_for(m: Model):
    with _lock:
        if m.key not in _cache:
            try:
                _cache[m.key] = _ADAPTERS[m.provider](m)
            except ImportError as e:
                raise ProviderError(f"{e}. The provider SDKs are in the failbench group: "
                                    "`uv run --group failbench so101-failbench ...`")
        return _cache[m.key]


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
                    timeout: float = 60, max_tokens: int = 4000, title: str = "so101") -> tuple[dict, float]:
    """One OpenRouter chat call with a JPEG and text, answered as JSON matching `schema`. Returns the
    answer and the call's cost in USD (OpenRouter's, or the model's prices without it). Raises on failure."""
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
    cost = float(cost) if cost is not None else m.cost(getattr(usage, "prompt_tokens", None),
                                                      getattr(usage, "completion_tokens", None))
    return json.loads(r.choices[0].message.content or "{}"), cost
