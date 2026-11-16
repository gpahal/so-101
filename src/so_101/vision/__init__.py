"""The vision judges `so101-failbench` benchmarks and the supervisor asks: one place for both.

Modules:
- `models`     the models, their provider, settings and prices
- `providers`  one adapter per provider (OpenRouter, Gemini, OpenAI, TypeSafe), and the supervisor's
               escalation call; the only module that imports a provider SDK, and only lazily
- `prompt`     the task description, moments, categories, the question and the answer schema
- `env`        `load_env`, the API keys from `.scratch/bench.env`

The provider SDKs are in the `failbench` dependency group: `uv sync --group failbench`.
"""
