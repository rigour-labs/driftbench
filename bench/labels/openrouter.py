"""One chat completion through OpenRouter, with the cost OpenRouter reports for it.

Model pre-labels use a model outside the Claude family (both paid entrants
run on Claude), reached through OpenRouter because it reports the real cost
of every call. The key is read from the environment when a call is made; it
is never logged, written or included in an error.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
KEY_ENDPOINT = "https://openrouter.ai/api/v1/key"
KEY_NAME = "OPENROUTER_API_KEY"
TIMEOUT_S = 120

Transport = Callable[[dict], dict]


class OpenRouterError(RuntimeError):
    pass


def post(body: dict) -> dict:
    """Send one request body to OpenRouter and return the decoded JSON response."""
    key = os.environ.get(KEY_NAME)
    if not key:
        raise OpenRouterError(f"{KEY_NAME} is not set")
    request = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise OpenRouterError(f"OpenRouter answered HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise OpenRouterError(f"OpenRouter call failed: {type(exc).__name__}") from None


def key_usage() -> float:
    """The key's cumulative billed usage in USD, as OpenRouter reports it; OpenRouterError if it can't be read.

    A GET with no body: urlopen's second positional parameter is the request body, so the timeout is passed
    by keyword."""
    key = os.environ.get(KEY_NAME)
    if not key:
        raise OpenRouterError(f"{KEY_NAME} is not set")
    request = urllib.request.Request(KEY_ENDPOINT, method="GET", headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            usage = (json.loads(response.read().decode("utf-8")).get("data") or {}).get("usage")
    except (OSError, ValueError, TypeError, AttributeError) as exc:  # URLError and timeouts are OSErrors
        raise OpenRouterError(f"reading the key's usage failed: {type(exc).__name__}") from None
    if not isinstance(usage, (int, float)):
        raise OpenRouterError("OpenRouter's key response has no numeric data.usage")
    return float(usage)


def chat(model: str, messages: list[dict], max_tokens: int, transport: Transport = post) -> dict:
    """{content, cost_usd, served_model}; cost_usd is None when OpenRouter reported none."""
    reply = transport({"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                       "usage": {"include": True}})
    choices = reply.get("choices") or []
    content = (choices[0].get("message") or {}).get("content") if choices else None
    cost = (reply.get("usage") or {}).get("cost")
    return {"content": content if isinstance(content, str) else None,
            "cost_usd": float(cost) if isinstance(cost, (int, float)) else None,
            "served_model": reply.get("model")}
