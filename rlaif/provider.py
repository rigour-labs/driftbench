"""Provider-agnostic LLM teacher for RLAIF pipeline.

Works with any OpenAI SDK-compatible provider via litellm routing.
Configure via env vars (MODEL_PROVIDER, MODEL_NAME, API_KEY, API_BASE)
or CLI args (--provider, --model-name, --api-key, --api-base).
"""

import os
import json
import logging
from typing import Dict, List

import litellm

logger = logging.getLogger("rlaif.provider")

# Provider -> env var mapping for API keys (litellm convention)
PROVIDER_KEY_ENV_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHERAI_API_KEY",
    "together_ai": "TOGETHERAI_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "fireworks_ai": "FIREWORKS_AI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "ollama": None,
}

# Defaults from env vars
DEFAULT_PROVIDER = os.environ.get("MODEL_PROVIDER", "anthropic")
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "claude-sonnet-4-20250514")
DEFAULT_API_KEY = os.environ.get("API_KEY", "")
DEFAULT_API_BASE = os.environ.get("API_BASE", "")
DEFAULT_TEACHER_MODEL = f"{DEFAULT_PROVIDER}/{DEFAULT_MODEL_NAME}"

# Teacher system prompt (same as rigour-core deep analysis)
TEACHER_SYSTEM_PROMPT = """You are an expert code reviewer performing deep quality analysis.
You receive AST-extracted facts about source files and must identify quality issues.

RULES:
1. ONLY report issues verifiable from the provided facts. Do NOT hallucinate.
2. Every finding MUST reference a real file, class, struct, or function from the facts.
3. Be specific: include file paths, entity names, line counts.
4. Assign confidence scores honestly: 0.9+ only for certain issues.
5. Respond ONLY with valid JSON. No text outside JSON.
6. Report ALL issues you can identify (aim for 5-15 per batch).

OUTPUT FORMAT:
{
  "findings": [
    {
      "category": "string",
      "severity": "critical|high|medium|low|info",
      "file": "string (exact path from facts)",
      "line": null,
      "description": "string (reference specific entities)",
      "suggestion": "string (actionable fix)",
      "confidence": 0.0-1.0
    }
  ]
}

CATEGORIES: srp_violation, ocp_violation, lsp_violation, isp_violation, dip_violation,
god_class, god_function, feature_envy, shotgun_surgery, long_params, data_clump,
inappropriate_intimacy, primitive_obsession, lazy_class, speculative_generality,
refused_bequest, dry_violation, copy_paste_code, error_inconsistency, empty_catch,
error_swallowing, missing_error_check, panic_in_library, race_condition, goroutine_leak,
missing_context, channel_misuse, mutex_scope, test_quality, test_coupling, missing_test,
test_duplication, long_file, magic_number, resource_leak, complex_conditional,
architecture, circular_dependency, package_cohesion, api_design, missing_abstraction,
language_idiom, naming_convention, dead_code, code_smell, performance, hardcoded_config"""


def setup_provider(
    provider: str, api_key: str = "", api_base: str = ""
):
    """Configure litellm for the given provider."""
    if api_key:
        env_var = PROVIDER_KEY_ENV_MAP.get(provider)
        if env_var:
            os.environ[env_var] = api_key
        os.environ["API_KEY"] = api_key

    if api_base:
        os.environ["API_BASE"] = api_base
        if provider in ("openai", "custom"):
            os.environ["OPENAI_API_BASE"] = api_base

    if provider not in ("ollama", "ollama_chat", "custom"):
        env_var = PROVIDER_KEY_ENV_MAP.get(
            provider, f"{provider.upper()}_API_KEY"
        )
        if env_var and not os.environ.get(env_var):
            if not os.environ.get("API_KEY") and not api_key:
                logger.warning(
                    f"No API key for '{provider}'. "
                    f"Set {env_var} or pass --api-key."
                )


def build_model_string(provider: str, model_name: str) -> str:
    """Build litellm model string from provider + model name."""
    if "/" in model_name and provider in model_name.split("/")[0]:
        return model_name
    if provider in ("ollama", "ollama_chat"):
        return f"ollama_chat/{model_name}"
    return f"{provider}/{model_name}"


def call_teacher(
    facts_prompt: str,
    model: str = DEFAULT_TEACHER_MODEL,
    batch_index: int = 0,
    api_base: str = "",
) -> List[Dict]:
    """Send facts to teacher model and get findings back."""
    user_prompt = (
        "Analyze the following codebase facts and "
        "identify ALL quality issues:\n\n" + facts_prompt
    )

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }

    provider = model.split("/")[0] if "/" in model else "unknown"
    if provider not in ("ollama", "ollama_chat"):
        kwargs["response_format"] = {"type": "json_object"}

    if api_base:
        kwargs["api_base"] = api_base

    try:
        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content
        content = _strip_code_fences(content)
        data = json.loads(content)
        findings = data.get("findings", [])
        logger.info(
            f"Teacher ({model}) returned {len(findings)} "
            f"findings (batch {batch_index})"
        )
        return findings
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse teacher JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"Teacher call failed ({model}): {e}")
        return []


def _strip_code_fences(content: str) -> str:
    """Strip markdown code fences from model output."""
    text = content.strip()
    if not text.startswith("```"):
        return text
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
