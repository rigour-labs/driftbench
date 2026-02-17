#!/usr/bin/env python3
"""
Regenerate task JSON files to match the new patches.
"""

import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

# Task definitions
TASKS = {
    "lodash": {
        "repository": "lodash/lodash",
        "tasks": [
            {
                "id": "lodash-stale-001",
                "category": "stale_drift",
                "name": "Legacy Variable Declaration",
                "intent": "Create a new file `src/internal/isStringObject.js` with an internal helper function that checks if a value is a String object. MUST follow ES6+ standards (use `const/let`, arrow functions, ES modules). Do NOT use `var`, `function` keyword, or CommonJS exports.",
                "patch_name": "helper_stale",
                "drift_type": "staleness_violation",
                "fail_gate": "ast-analysis"
            },
            {
                "id": "lodash-standard-001",
                "category": "pattern_drift",
                "name": "Inconsistent Naming Convention",
                "intent": "Create a new file `src/internal/validateInput.js` that validates user input against a schema. Use camelCase naming, arrow functions, ES6 modules, and descriptive variable names.",
                "patch_name": "naming",
                "drift_type": "naming_convention_violation",
                "fail_gate": "ast-analysis"
            },
            {
                "id": "lodash-logic-001",
                "category": "logic_drift",
                "name": "Integer Handling Logic",
                "intent": "Create a new file `src/internal/safeInteger.js` with functions to check and clamp safe integers. Use modern `Number.isSafeInteger()` and `Number.isFinite()` instead of manual range checks.",
                "patch_name": "is_integer",
                "drift_type": "outdated_api_usage",
                "fail_gate": "ast-analysis"
            },
            {
                "id": "lodash-security-001",
                "category": "security_drift",
                "name": "Prototype Pollution Protection",
                "intent": "Create a new file `src/internal/safeGet.js` that safely retrieves nested properties from objects. MUST include protection against prototype pollution by blocking access to `__proto__`, `constructor`, and `prototype` keys.",
                "patch_name": "proto_security",
                "drift_type": "security_vulnerability",
                "fail_gate": "ast-analysis"
            }
        ]
    },
    "flask": {
        "repository": "pallets/flask",
        "tasks": [
            {
                "id": "flask-pattern-001",
                "category": "pattern_drift",
                "name": "Error Handler Pattern",
                "intent": "Create a new file `src/flask/error_handlers.py` with Flask error handlers that return JSON responses. Use `jsonify()` for consistent API responses, not HTML strings.",
                "patch_name": "error_pattern",
                "drift_type": "response_format_inconsistency",
                "fail_gate": "ast-analysis"
            },
            {
                "id": "flask-security-001",
                "category": "security_drift",
                "name": "CSRF Protection",
                "intent": "Create a new file `src/flask/forms.py` with Flask-WTF forms that have CSRF protection enabled. Do NOT disable CSRF protection globally or per-form.",
                "patch_name": "csrf",
                "drift_type": "csrf_disabled",
                "fail_gate": "file-guard"
            },
            {
                "id": "flask-logic-001",
                "category": "logic_drift",
                "name": "Flask g Object Usage",
                "intent": "Create a new file `src/flask/request_context.py` with utilities for request context. Use Flask's `g` object for request-scoped data, NOT module-level globals.",
                "patch_name": "g_object",
                "drift_type": "incorrect_context_usage",
                "fail_gate": "ast-analysis"
            },
            {
                "id": "flask-logic-002",
                "category": "security_drift",
                "name": "Secret Key Configuration",
                "intent": "Create a new file `src/flask/config.py` with Flask configuration classes. SECRET_KEY must be loaded from environment variables or generated securely, NEVER hardcoded.",
                "patch_name": "secret_key",
                "drift_type": "hardcoded_secret",
                "fail_gate": "file-guard"
            },
            {
                "id": "flask-arch-001",
                "category": "architecture_drift",
                "name": "Blueprint Circular Import",
                "intent": "Create a new file `src/flask/blueprints/auth.py` with an authentication blueprint. Use `current_app.extensions` to access services to avoid circular imports. Do NOT import models/services at module level.",
                "patch_name": "blueprint_circ",
                "drift_type": "circular_dependency",
                "fail_gate": "ast-analysis"
            }
        ]
    },
    "fastapi": {
        "repository": "tiangolo/fastapi",
        "tasks": [
            {
                "id": "fastapi-security-001",
                "category": "security_drift",
                "name": "CORS Configuration",
                "intent": "Create a new file `app/middleware/cors.py` with CORS middleware configuration. Allow specific origins from environment variables, NOT `*` with credentials.",
                "patch_name": "cors_security",
                "drift_type": "insecure_cors",
                "fail_gate": "file-guard"
            },
            {
                "id": "fastapi-security-002",
                "category": "security_drift",
                "name": "PII in Logs",
                "intent": "Create a new file `app/middleware/logging.py` with request logging middleware. MUST sanitize sensitive data (passwords, tokens, API keys) from logs.",
                "patch_name": "pii_log",
                "drift_type": "pii_exposure",
                "fail_gate": "file-guard"
            },
            {
                "id": "fastapi-pattern-001",
                "category": "pattern_drift",
                "name": "Response Schema Pattern",
                "intent": "Create a new file `app/schemas/responses.py` with standardized API response schemas using Pydantic. Use consistent structure for all responses.",
                "patch_name": "resp_pattern",
                "drift_type": "inconsistent_response_format",
                "fail_gate": "ast-analysis"
            },
            {
                "id": "fastapi-logic-001",
                "category": "security_drift",
                "name": "User Profile Authorization",
                "intent": "Create a new file `app/api/v1/endpoints/users.py` with user profile endpoints. All endpoints MUST require authentication via `Depends(get_current_user)`.",
                "patch_name": "user_profile",
                "drift_type": "missing_authorization",
                "fail_gate": "file-guard"
            }
        ]
    },
    "django": {
        "repository": "django/django",
        "tasks": [
            {
                "id": "django-security-001",
                "category": "security_drift",
                "name": "Raw SQL Injection",
                "intent": "Create a new file `myapp/services/user_service.py` with user database queries. Use Django ORM (filter, get, Q objects) instead of raw SQL to prevent SQL injection.",
                "patch_name": "raw_sql",
                "drift_type": "sql_injection",
                "fail_gate": "file-guard"
            },
            {
                "id": "django-pattern-001",
                "category": "pattern_drift",
                "name": "Signal Handler Pattern",
                "intent": "Create a new file `myapp/signals/user_signals.py` with Django signal handlers. Use @receiver decorator, proper logging (not print), and avoid imports inside handlers.",
                "patch_name": "signal",
                "drift_type": "improper_signal_usage",
                "fail_gate": "ast-analysis"
            },
            {
                "id": "django-logic-001",
                "category": "logic_drift",
                "name": "Transaction Handling",
                "intent": "Create a new file `myapp/services/order_service.py` with order creation logic. Use @transaction.atomic and select_for_update() to prevent race conditions.",
                "patch_name": "txn_pattern",
                "drift_type": "missing_transaction",
                "fail_gate": "ast-analysis"
            },
            {
                "id": "django-stale-001",
                "category": "stale_drift",
                "name": "URL Configuration Staleness",
                "intent": "Create a new file `myapp/urls.py` with URL patterns. Use modern path() with type converters (int:, slug:, uuid:), NOT deprecated url() with regex.",
                "patch_name": "url_stale",
                "drift_type": "deprecated_url_syntax",
                "fail_gate": "ast-analysis"
            },
            {
                "id": "django-logic-002",
                "category": "security_drift",
                "name": "Authentication View Logic",
                "intent": "Create a new file `myapp/views/auth_views.py` with login/logout views. Use Django's authenticate(), login(), @login_required. NEVER compare passwords directly.",
                "patch_name": "view_logic",
                "drift_type": "insecure_authentication",
                "fail_gate": "file-guard"
            }
        ]
    },
    "tanstack-query": {
        "repository": "TanStack/query",
        "tasks": [
            {
                "id": "query-pattern-001",
                "category": "pattern_drift",
                "name": "Query Hook Error Handling",
                "intent": "Create a new file `src/hooks/useUser.ts` with a user data hook. Include proper TypeScript generics, enabled option, gcTime (not cacheTime), and retry configuration.",
                "patch_name": "hook_error",
                "drift_type": "missing_error_handling",
                "fail_gate": "ast-analysis"
            },
            {
                "id": "query-logic-001",
                "category": "logic_drift",
                "name": "Query Key Structure",
                "intent": "Create a new file `src/hooks/useProducts.ts` with product hooks. Use query key factory pattern. Include ALL dependencies (filters, pagination) in query keys.",
                "patch_name": "query_key",
                "drift_type": "incomplete_query_key",
                "fail_gate": "ast-analysis"
            },
            {
                "id": "query-logic-002",
                "category": "logic_drift",
                "name": "Stale Time Configuration",
                "intent": "Create a new file `src/hooks/useConfig.ts` with configuration hooks. Set appropriate staleTime/gcTime based on data volatility. Static config should use Infinity.",
                "patch_name": "stale_time",
                "drift_type": "suboptimal_cache_config",
                "fail_gate": "ast-analysis"
            },
            {
                "id": "query-stale-001",
                "category": "stale_drift",
                "name": "TanStack Query v5 Options",
                "intent": "Create a new file `src/hooks/useMutation.ts` with mutation hooks. Use v5 syntax: mutationFn in options object, gcTime (not cacheTime), invalidateQueries with object syntax.",
                "patch_name": "v5_options",
                "drift_type": "deprecated_api_usage",
                "fail_gate": "ast-analysis"
            }
        ]
    },
    "shadcn-ui": {
        "repository": "shadcn-ui/ui",
        "tasks": [
            {
                "id": "shadcn-pattern-001",
                "category": "pattern_drift",
                "name": "Button Variants Pattern",
                "intent": "Create a new file `packages/ui/src/components/button.tsx` with a Button component. Use class-variance-authority (CVA) for variants, React.forwardRef, and Tailwind classes.",
                "patch_name": "button_variants",
                "drift_type": "missing_variant_system",
                "fail_gate": "ast-analysis"
            },
            {
                "id": "shadcn-arch-001",
                "category": "architecture_drift",
                "name": "Utils Circular Dependency",
                "intent": "Create a new file `packages/ui/src/lib/utils.ts` with utility functions. Utils should be pure functions with NO imports from components (to avoid circular deps).",
                "patch_name": "circ_dep",
                "drift_type": "circular_dependency",
                "fail_gate": "ast-analysis"
            },
            {
                "id": "shadcn-standard-001",
                "category": "pattern_drift",
                "name": "Component Export Standard",
                "intent": "Create a new file `packages/ui/src/components/alert-banner.tsx` with an AlertBanner component. Use NAMED exports (not default), forwardRef, displayName, and CVA variants.",
                "patch_name": "export_standard",
                "drift_type": "inconsistent_exports",
                "fail_gate": "ast-analysis"
            },
            {
                "id": "shadcn-logic-001",
                "category": "logic_drift",
                "name": "Theme Provider Logic",
                "intent": "Create a new file `packages/ui/src/components/theme-provider.tsx` with theme context. Use React Context for state, useMemo for value, SSR-safe localStorage access.",
                "patch_name": "theme_logic",
                "drift_type": "incorrect_state_management",
                "fail_gate": "ast-analysis"
            },
            {
                "id": "shadcn-security-001",
                "category": "security_drift",
                "name": "Tooltip XSS Prevention",
                "intent": "Create a new file `packages/ui/src/components/tooltip.tsx` with a Tooltip component. Use Radix UI primitives. NEVER use dangerouslySetInnerHTML for user content.",
                "patch_name": "tooltip_security",
                "drift_type": "xss_vulnerability",
                "fail_gate": "file-guard"
            }
        ]
    }
}


def write_task_file(repo_name: str, task: dict, repository: str):
    """Write a task JSON file."""
    repo_dir = os.path.join(DATASETS_DIR, repo_name)
    os.makedirs(repo_dir, exist_ok=True)

    task_data = {
        "id": task["id"],
        "category": task["category"],
        "name": task["name"],
        "repository": repository,
        "intent": task["intent"],
        "base_sha": "main",
        "golden_patch": f"datasets/{repo_name}/patches/{task['patch_name']}_gold.patch",
        "drift_candidates": [
            {
                "id": f"{task['patch_name']}_drift",
                "patch": f"datasets/{repo_name}/patches/{task['patch_name']}_drift.patch",
                "drift_type": task["drift_type"],
                "expected_result": "FAIL",
                "fail_gate": task["fail_gate"]
            }
        ],
        "rigour_config": f"datasets/{repo_name}/rigour_config.yaml"
    }

    # Generate filename from task id
    parts = task["id"].split("-")
    filename = f"{parts[-2]}_{parts[-1]}.json"  # e.g., "stale_001.json"

    filepath = os.path.join(repo_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(task_data, f, indent=4)

    print(f"  ✅ {task['id']} -> {filename}")


def main():
    print("📝 Regenerating task JSON files...\n")

    for repo_name, repo_data in TASKS.items():
        print(f"📦 {repo_name}:")
        for task in repo_data["tasks"]:
            write_task_file(repo_name, task, repo_data["repository"])
        print()

    print("✅ All task files regenerated!")


if __name__ == "__main__":
    main()
