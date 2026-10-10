"""
Conntrail live demo service.

A tiny FastAPI service that runs the *real* Conntrail SDK — contrast
generation + divergence analysis — against a fixed routing decision, using
free-tier LLM providers. It powers the "Try it live" mode on the website.

Two modes:
  - free trial: the server's own free-tier key (Groq or OpenRouter), capped
    per-IP and per-day with response caching;
  - bring-your-own-key: the visitor supplies a free key, used only for that
    request and never stored.
"""
