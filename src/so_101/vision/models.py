"""The vision models `so101-failbench` compares and the supervisor asks, how each is called and what it costs.

Prices are USD per million tokens, from each provider's pricing page (OpenRouter's catalog for the
OpenRouter models) in November 2026; check them before trusting the cost columns. `estimate` is a rough
cost per sample used only before a model's first answer, to keep a run under its cap; after that the
runner uses the model's real average. OpenRouter reports each call's actual cost, which the report uses.

Chat models run at a low reasoning effort by default: a failure check in the control loop has to answer
in seconds, not minutes. `claude-opus-5.5-high` is the same model at high effort, to show what thinking
longer buys.
"""

import os
from dataclasses import dataclass, field

# Providers and the variable holding each one's key.
OPENROUTER = "openrouter"
GEMINI = "gemini"
GEMINI_DECISIONS = "gemini-decisions"
OPENAI_DECISIONS = "openai-decisions"
TYPESAFE = "typesafe"
KEYS = {OPENROUTER: "OPENROUTER_API_KEY", GEMINI: "GEMINI_API_KEY", GEMINI_DECISIONS: "GEMINI_API_KEY",
        OPENAI_DECISIONS: "OPENAI_API_KEY", TYPESAFE: "TYPESAFE_API_KEY"}


@dataclass(frozen=True)
class Model:
    key: str  # the name in reports and on the command line
    provider: str
    model_id: str
    price_in: float
    price_out: float
    group: str  # claude, gpt, gemini, robotics, decision
    effort: str | None = None
    estimate: float = 0.01
    options: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def env(self) -> str:
        return KEYS[self.provider]

    def has_key(self) -> bool:
        return bool(os.environ.get(self.env))

    def adapter(self):
        from so_101.vision import providers
        return providers.adapter_for(self)

    def cost(self, tokens_in: int | None, tokens_out: int | None) -> float:
        return ((tokens_in or 0) * self.price_in + (tokens_out or 0) * self.price_out) / 1e6


def _chat(key, model_id, pin, pout, group, effort="low", out_tokens=700, in_tokens=1800, **options):
    return Model(key, OPENROUTER, model_id, pin, pout, group, effort,
                 estimate=(in_tokens * pin + out_tokens * pout) / 1e6, options=options)


_LIST = [
    # Claude, through OpenRouter
    _chat("claude-opus-5.5", "anthropic/claude-opus-5.5", 4.00, 20.00, "claude"),
    _chat("claude-opus-5.5-high", "anthropic/claude-opus-5.5", 4.00, 20.00, "claude", effort="high", out_tokens=3000),
    _chat("claude-sonnet-5.5", "anthropic/claude-sonnet-5.5", 2.00, 10.00, "claude"),
    _chat("claude-haiku-5.5", "anthropic/claude-haiku-5.5", 0.10, 0.50, "claude"),
    _chat("claude-fable-5.2", "anthropic/claude-fable-5.2", 8.00, 40.00, "claude"),
    # GPT, through OpenRouter
    _chat("gpt-6-astra", "openai/gpt-6-astra", 10.00, 50.00, "gpt", in_tokens=1500),
    _chat("gpt-6.1-sol", "openai/gpt-6.1-sol", 2.00, 10.00, "gpt", in_tokens=1500),
    _chat("gpt-6.1-terra", "openai/gpt-6.1-terra", 1.00, 5.00, "gpt", in_tokens=1500),
    _chat("gpt-6-luna", "openai/gpt-6-luna", 0.10, 0.50, "gpt", in_tokens=1500),
    # Gemini, through OpenRouter. Gemini counts up to ~1,120 tokens per image by default.
    _chat("gemini-4-carbon", "google/gemini-4-carbon", 2.00, 10.00, "gemini", in_tokens=3000),
    _chat("gemini-4-argon", "google/gemini-4-argon", 2.00, 10.00, "gemini", in_tokens=3000),
    _chat("gemini-3.8-flash", "google/gemini-3.8-flash", 0.83, 4.12, "gemini", in_tokens=3000),
    _chat("gemini-3.5-flash-lite", "google/gemini-3.5-flash-lite", 0.30, 2.50, "gemini", effort="minimal", in_tokens=3000),
    # Gemini Robotics-ER 2, directly (not on OpenRouter). Google suggests medium thinking for latency.
    Model("gemini-robotics-er-2", GEMINI, "gemini-robotics-er-2-preview", 1.00, 5.00, "robotics", "medium", estimate=0.006),
    # Decision models: no text out, a probability per question, input tokens billed only.
    Model("gemini-decision-1", GEMINI_DECISIONS, "gemini-decision-1", 0.10, 0.0, "decision", estimate=0.0005),
    Model("openai-decisions-luna", OPENAI_DECISIONS, "gpt-6-luna", 0.10, 0.0, "decision", estimate=0.0005),
    Model("jev", TYPESAFE, "jev-latest", 0.042, 0.0, "decision", estimate=0.0003),
]
MODELS = {m.key: m for m in _LIST}
