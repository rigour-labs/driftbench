#!/usr/bin/env python3
"""
Regenerate all dataset patches to be production-quality.

Strategy:
- Create NEW files instead of modifying existing ones (reliable patching)
- Use realistic paths that match repo conventions
- Golden patches show correct modern patterns
- Drift patches show anti-patterns that rigour should detect
"""

import os
import json

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")


def create_new_file_patch(file_path: str, content: str) -> str:
    """Create a patch that adds a new file."""
    lines = content.strip().split('\n')
    patch_lines = [
        f"--- /dev/null",
        f"+++ b/{file_path}",
        f"@@ -0,0 +1,{len(lines)} @@"
    ]
    for line in lines:
        patch_lines.append(f"+{line}")
    return '\n'.join(patch_lines) + '\n'


# ============================================================================
# LODASH PATCHES (JavaScript)
# ============================================================================

LODASH_PATCHES = {
    "helper_stale": {
        "gold": {
            "path": "src/internal/isStringObject.js",
            "content": '''/**
 * Checks if `value` is classified as a `String` object.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a string object, else `false`.
 */
const isStringObject = (value) => {
  if (value == null) return false;
  return Object.prototype.toString.call(value) === '[object String]';
};

export default isStringObject;'''
        },
        "drift": {
            "path": "src/internal/isStringObject.js",
            "content": '''/**
 * Checks if `value` is classified as a `String` object.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a string object, else `false`.
 */
var isStringObject = function(value) {
  if (value == null) return false;
  return Object.prototype.toString.call(value) === '[object String]';
};

module.exports = isStringObject;'''
        }
    },
    "naming": {
        "gold": {
            "path": "src/internal/validateInput.js",
            "content": '''/**
 * Validates user input according to schema.
 * @param {Object} input - The input to validate
 * @param {Object} schema - Validation schema
 * @returns {boolean} Whether input is valid
 */
const validateInput = (input, schema) => {
  if (!input || !schema) return false;

  return Object.keys(schema).every((key) => {
    const rule = schema[key];
    const value = input[key];

    if (rule.required && value === undefined) return false;
    if (rule.type && typeof value !== rule.type) return false;

    return true;
  });
};

export default validateInput;'''
        },
        "drift": {
            "path": "src/internal/validateInput.js",
            "content": '''/**
 * Validates user input according to schema.
 */
var validate_input = function(i, s) {
  if (!i || !s) return false;

  var keys = Object.keys(s);
  for (var x = 0; x < keys.length; x++) {
    var r = s[keys[x]];
    var v = i[keys[x]];

    if (r.required && v === undefined) return false;
    if (r.type && typeof v !== r.type) return false;
  }
  return true;
};

module.exports = validate_input;'''
        }
    },
    "is_integer": {
        "gold": {
            "path": "src/internal/safeInteger.js",
            "content": '''/**
 * Checks if value is a safe integer.
 * @param {*} value - The value to check
 * @returns {boolean} Whether value is a safe integer
 */
const isSafeInteger = (value) => {
  return Number.isSafeInteger(value);
};

/**
 * Clamps value to safe integer range.
 * @param {number} value - The value to clamp
 * @returns {number} The clamped value
 */
const clampToSafeInteger = (value) => {
  if (!Number.isFinite(value)) return 0;
  return Math.max(
    Number.MIN_SAFE_INTEGER,
    Math.min(Number.MAX_SAFE_INTEGER, Math.trunc(value))
  );
};

export { isSafeInteger, clampToSafeInteger };'''
        },
        "drift": {
            "path": "src/internal/safeInteger.js",
            "content": '''/**
 * Checks if value is a safe integer.
 */
var isSafeInteger = function(value) {
  // Old polyfill approach instead of Number.isSafeInteger
  return typeof value === 'number' &&
         value === Math.floor(value) &&
         value >= -9007199254740991 &&
         value <= 9007199254740991;
};

var clampToSafeInteger = function(value) {
  if (value !== value || value === Infinity || value === -Infinity) return 0;
  var truncated = value > 0 ? Math.floor(value) : Math.ceil(value);
  return Math.max(-9007199254740991, Math.min(9007199254740991, truncated));
};

module.exports = { isSafeInteger: isSafeInteger, clampToSafeInteger: clampToSafeInteger };'''
        }
    },
    "proto_security": {
        "gold": {
            "path": "src/internal/safeGet.js",
            "content": '''/**
 * Safely gets a nested property from an object.
 * Protects against prototype pollution attacks.
 *
 * @param {Object} object - The object to query
 * @param {string} path - The path of the property to get
 * @returns {*} The resolved value
 */
const FORBIDDEN_KEYS = new Set(['__proto__', 'constructor', 'prototype']);

const safeGet = (object, path) => {
  if (object == null) return undefined;

  const keys = path.split('.');
  let result = object;

  for (const key of keys) {
    if (FORBIDDEN_KEYS.has(key)) {
      return undefined;
    }
    if (result == null) return undefined;
    result = result[key];
  }

  return result;
};

export default safeGet;'''
        },
        "drift": {
            "path": "src/internal/safeGet.js",
            "content": '''/**
 * Gets a nested property from an object.
 */
var safeGet = function(object, path) {
  if (object == null) return undefined;

  var keys = path.split('.');
  var result = object;

  // WARNING: No prototype pollution protection!
  for (var i = 0; i < keys.length; i++) {
    if (result == null) return undefined;
    result = result[keys[i]];
  }

  return result;
};

module.exports = safeGet;'''
        }
    }
}


# ============================================================================
# FLASK PATCHES (Python)
# ============================================================================

FLASK_PATCHES = {
    "error_pattern": {
        "gold": {
            "path": "src/flask/error_handlers.py",
            "content": '''"""Custom error handlers for the Flask application."""
from flask import jsonify


def register_error_handlers(app):
    """Register error handlers with the Flask app."""

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "not_found", "message": str(error)}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "internal_error", "message": "An unexpected error occurred"}), 500

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"error": "bad_request", "message": str(error)}), 400'''
        },
        "drift": {
            "path": "src/flask/error_handlers.py",
            "content": '''"""Custom error handlers for the Flask application."""


def register_error_handlers(app):
    """Register error handlers with the Flask app."""

    @app.errorhandler(404)
    def not_found(error):
        # Anti-pattern: returning HTML string instead of JSON
        return "<h1>404 Not Found</h1><p>" + str(error) + "</p>", 404

    @app.errorhandler(500)
    def internal_error(error):
        # Anti-pattern: exposing internal error details
        return "<h1>500 Error</h1><pre>" + str(error) + "</pre>", 500

    @app.errorhandler(400)
    def bad_request(error):
        return "<h1>Bad Request</h1>", 400'''
        }
    },
    "csrf": {
        "gold": {
            "path": "src/flask/forms.py",
            "content": '''"""Form handling with CSRF protection."""
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Email


csrf = CSRFProtect()


class ContactForm(FlaskForm):
    """Contact form with CSRF protection enabled by default."""
    name = StringField('Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    message = StringField('Message', validators=[DataRequired()])
    submit = SubmitField('Send')


def init_csrf(app):
    """Initialize CSRF protection for the app."""
    csrf.init_app(app)'''
        },
        "drift": {
            "path": "src/flask/forms.py",
            "content": '''"""Form handling - CSRF disabled for convenience."""
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired


class ContactForm(FlaskForm):
    """Contact form with CSRF disabled."""

    class Meta:
        csrf = False  # SECURITY DRIFT: CSRF protection disabled!

    name = StringField('Name', validators=[DataRequired()])
    email = StringField('Email')
    message = StringField('Message')
    submit = SubmitField('Send')


def init_csrf(app):
    """CSRF initialization skipped."""
    app.config['WTF_CSRF_ENABLED'] = False  # SECURITY DRIFT!
    pass'''
        }
    },
    "g_object": {
        "gold": {
            "path": "src/flask/request_context.py",
            "content": '''"""Request context utilities using Flask's g object properly."""
from flask import g, current_app
from functools import wraps


def get_db():
    """Get database connection from g object."""
    if 'db' not in g:
        g.db = current_app.config['DATABASE'].connect()
    return g.db


def close_db(e=None):
    """Close database connection stored in g."""
    db = g.pop('db', None)
    if db is not None:
        db.close()


def with_db(f):
    """Decorator to ensure database connection is available."""
    @wraps(f)
    def decorated(*args, **kwargs):
        g.db = get_db()
        try:
            return f(*args, **kwargs)
        finally:
            close_db()
    return decorated'''
        },
        "drift": {
            "path": "src/flask/request_context.py",
            "content": '''"""Request context utilities - using globals incorrectly."""

# LOGIC DRIFT: Using module-level globals instead of Flask's g
_db_connection = None
_current_user = None


def get_db():
    """Get database connection from module global."""
    global _db_connection
    if _db_connection is None:
        from flask import current_app
        _db_connection = current_app.config['DATABASE'].connect()
    return _db_connection


def close_db(e=None):
    """Close database connection."""
    global _db_connection
    if _db_connection is not None:
        _db_connection.close()
        _db_connection = None


def with_db(f):
    """Decorator for database access."""
    def decorated(*args, **kwargs):
        global _db_connection
        _db_connection = get_db()
        return f(*args, **kwargs)
    return decorated'''
        }
    },
    "secret_key": {
        "gold": {
            "path": "src/flask/config.py",
            "content": '''"""Application configuration with secure defaults."""
import os
import secrets


class Config:
    """Base configuration with secure defaults."""

    # Generate a secure random secret key
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(32)

    # Security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    SESSION_COOKIE_SECURE = False


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False

    @classmethod
    def init_app(cls, app):
        # Ensure SECRET_KEY is set in production
        assert cls.SECRET_KEY != 'dev', "SECRET_KEY must be set in production"'''
        },
        "drift": {
            "path": "src/flask/config.py",
            "content": '''"""Application configuration."""


class Config:
    """Base configuration."""

    # SECURITY DRIFT: Hardcoded secret key!
    SECRET_KEY = 'super-secret-key-12345'

    # SECURITY DRIFT: Insecure session settings
    SESSION_COOKIE_SECURE = False
    SESSION_COOKIE_HTTPONLY = False

    # Database
    SQLALCHEMY_DATABASE_URI = 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = True


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration - same as dev."""
    DEBUG = True  # LOGIC DRIFT: Debug enabled in production!'''
        }
    },
    "blueprint_circ": {
        "gold": {
            "path": "src/flask/blueprints/auth.py",
            "content": '''"""Authentication blueprint with proper imports."""
from flask import Blueprint, request, jsonify, current_app
from werkzeug.security import generate_password_hash, check_password_hash


bp = Blueprint('auth', __name__, url_prefix='/auth')


@bp.route('/login', methods=['POST'])
def login():
    """Handle user login."""
    data = request.get_json()

    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'error': 'Missing credentials'}), 400

    # Get user service from app context to avoid circular imports
    user_service = current_app.extensions.get('user_service')
    if not user_service:
        return jsonify({'error': 'Service unavailable'}), 503

    user = user_service.authenticate(data['username'], data['password'])
    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401

    return jsonify({'message': 'Login successful', 'user_id': user.id})


@bp.route('/register', methods=['POST'])
def register():
    """Handle user registration."""
    data = request.get_json()

    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'error': 'Missing required fields'}), 400

    user_service = current_app.extensions.get('user_service')
    user = user_service.create_user(
        username=data['username'],
        password_hash=generate_password_hash(data['password'])
    )

    return jsonify({'message': 'User created', 'user_id': user.id}), 201'''
        },
        "drift": {
            "path": "src/flask/blueprints/auth.py",
            "content": '''"""Authentication blueprint with circular import issues."""
from flask import Blueprint, request, jsonify
# ARCHITECTURE DRIFT: Direct import creates circular dependency
from src.flask.models import User
from src.flask.services import UserService


bp = Blueprint('auth', __name__, url_prefix='/auth')

# ARCHITECTURE DRIFT: Module-level instantiation
user_service = UserService()


@bp.route('/login', methods=['POST'])
def login():
    """Handle user login."""
    data = request.get_json()

    # LOGIC DRIFT: No input validation
    user = User.query.filter_by(username=data['username']).first()

    # SECURITY DRIFT: Plain text password comparison
    if user and user.password == data['password']:
        return jsonify({'message': 'Login successful'})

    return jsonify({'error': 'Invalid'}), 401


@bp.route('/register', methods=['POST'])
def register():
    """Handle user registration."""
    data = request.get_json()

    # SECURITY DRIFT: Storing plain text password
    user = User(username=data['username'], password=data['password'])
    user.save()

    return jsonify({'message': 'User created'})'''
        }
    }
}


# ============================================================================
# FASTAPI PATCHES (Python)
# ============================================================================

FASTAPI_PATCHES = {
    "cors_security": {
        "gold": {
            "path": "app/middleware/cors.py",
            "content": '''"""CORS middleware configuration with secure defaults."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os


def configure_cors(app: FastAPI, allowed_origins: List[str] = None) -> None:
    """Configure CORS with secure defaults.

    Args:
        app: FastAPI application instance
        allowed_origins: List of allowed origins (defaults to env var or empty)
    """
    if allowed_origins is None:
        # Get from environment, default to empty (no CORS)
        origins_env = os.environ.get("CORS_ORIGINS", "")
        allowed_origins = [o.strip() for o in origins_env.split(",") if o.strip()]

    if not allowed_origins:
        return  # No CORS if no origins specified

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
        max_age=600,  # Cache preflight for 10 minutes
    )'''
        },
        "drift": {
            "path": "app/middleware/cors.py",
            "content": '''"""CORS middleware configuration."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def configure_cors(app: FastAPI) -> None:
    """Configure CORS - allowing everything for convenience."""

    # SECURITY DRIFT: Allow all origins!
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # DANGEROUS: Allows any origin
        allow_credentials=True,  # With credentials = security issue
        allow_methods=["*"],  # All methods
        allow_headers=["*"],  # All headers
    )'''
        }
    },
    "pii_log": {
        "gold": {
            "path": "app/middleware/logging.py",
            "content": '''"""Request logging middleware with PII protection."""
import logging
import re
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Set


logger = logging.getLogger(__name__)

# Fields that should never be logged
SENSITIVE_FIELDS: Set[str] = {
    'password', 'token', 'secret', 'api_key', 'apikey',
    'authorization', 'credit_card', 'ssn', 'social_security'
}

# Patterns to redact
REDACT_PATTERNS = [
    (re.compile(r'"password"\s*:\s*"[^"]*"'), '"password": "[REDACTED]"'),
    (re.compile(r'"token"\s*:\s*"[^"]*"'), '"token": "[REDACTED]"'),
    (re.compile(r'Bearer\s+\S+'), 'Bearer [REDACTED]'),
]


def sanitize_log_data(data: str) -> str:
    """Remove sensitive information from log data."""
    result = data
    for pattern, replacement in REDACT_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs requests with PII protection."""

    async def dispatch(self, request: Request, call_next):
        # Log request (sanitized)
        logger.info(
            "Request: %s %s",
            request.method,
            request.url.path  # Don't log query params (may contain tokens)
        )

        response = await call_next(request)

        logger.info(
            "Response: %s %s -> %d",
            request.method,
            request.url.path,
            response.status_code
        )

        return response'''
        },
        "drift": {
            "path": "app/middleware/logging.py",
            "content": '''"""Request logging middleware."""
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs all request details."""

    async def dispatch(self, request: Request, call_next):
        # SECURITY DRIFT: Logging full request including sensitive data!
        body = await request.body()
        logger.info(
            "Request: %s %s?%s Body: %s Headers: %s",
            request.method,
            request.url.path,
            request.url.query,  # May contain tokens/API keys
            body.decode(),  # May contain passwords
            dict(request.headers)  # Contains Authorization header
        )

        response = await call_next(request)

        logger.info("Response: %s", response.status_code)

        return response'''
        }
    },
    "resp_pattern": {
        "gold": {
            "path": "app/schemas/responses.py",
            "content": '''"""Standardized API response schemas."""
from pydantic import BaseModel
from typing import TypeVar, Generic, Optional, List


T = TypeVar('T')


class APIResponse(BaseModel, Generic[T]):
    """Standard API response wrapper."""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    message: Optional[str] = None


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response for list endpoints."""
    items: List[T]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[dict] = None'''
        },
        "drift": {
            "path": "app/schemas/responses.py",
            "content": '''"""API response handling."""
from typing import Any


def make_response(data: Any, status: str = "ok") -> dict:
    """Create API response - inconsistent format."""
    # PATTERN DRIFT: Inconsistent response structure
    if status == "ok":
        return {"result": data, "status": status}
    else:
        return {"err": data, "stat": status}


def error_response(msg: str) -> dict:
    """Create error response."""
    # PATTERN DRIFT: Different structure than success response
    return {"error_message": msg, "is_error": True}


def list_response(items: list) -> dict:
    """Create list response."""
    # PATTERN DRIFT: Yet another structure
    return {"data": {"items": items, "count": len(items)}}'''
        }
    },
    "user_profile": {
        "gold": {
            "path": "app/api/v1/endpoints/users.py",
            "content": '''"""User profile endpoints with proper authorization."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional

from app.core.deps import get_db, get_current_user
from app.models.user import User
from app.schemas.user import UserRead, UserUpdate


router = APIRouter()


@router.get("/me", response_model=UserRead)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user)
):
    """Get the current authenticated user's profile."""
    return current_user


@router.get("/{user_id}", response_model=UserRead)
async def get_user_by_id(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get user by ID (requires authentication)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


@router.put("/me", response_model=UserRead)
async def update_current_user(
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update the current user's profile."""
    for field, value in user_update.dict(exclude_unset=True).items():
        setattr(current_user, field, value)
    db.commit()
    db.refresh(current_user)
    return current_user'''
        },
        "drift": {
            "path": "app/api/v1/endpoints/users.py",
            "content": '''"""User profile endpoints."""
from fastapi import APIRouter, HTTPException
from app.models.user import User


router = APIRouter()


@router.get("/me")
async def get_current_user_profile(user_id: int):
    """Get user profile - no authentication!"""
    # SECURITY DRIFT: No authentication required
    user = User.query.get(user_id)
    return {"user": user.__dict__}


@router.get("/{user_id}")
async def get_user_by_id(user_id: int):
    """Get any user by ID - no authorization!"""
    # SECURITY DRIFT: Anyone can access any user's data
    user = User.query.get(user_id)
    if not user:
        raise HTTPException(status_code=404)
    # SECURITY DRIFT: Exposing all fields including sensitive ones
    return user.__dict__


@router.put("/{user_id}")
async def update_user(user_id: int, data: dict):
    """Update any user - no authorization!"""
    # SECURITY DRIFT: Anyone can update any user
    user = User.query.get(user_id)
    for k, v in data.items():
        setattr(user, k, v)  # SECURITY DRIFT: Mass assignment
    user.save()
    return {"updated": True}'''
        }
    }
}


# ============================================================================
# DJANGO PATCHES (Python)
# ============================================================================

DJANGO_PATCHES = {
    "raw_sql": {
        "gold": {
            "path": "myapp/services/user_service.py",
            "content": '''"""User service with safe database queries."""
from django.db.models import Q
from django.contrib.auth.models import User
from typing import List, Optional


class UserService:
    """Service for user-related database operations."""

    def search_users(self, query: str) -> List[User]:
        """Search users by username or email safely.

        Uses Django ORM to prevent SQL injection.
        """
        if not query or len(query) < 2:
            return []

        return User.objects.filter(
            Q(username__icontains=query) | Q(email__icontains=query)
        ).select_related('profile')[:100]

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID safely."""
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None

    def get_active_users(self, limit: int = 50) -> List[User]:
        """Get active users using ORM."""
        return User.objects.filter(
            is_active=True
        ).order_by('-last_login')[:limit]'''
        },
        "drift": {
            "path": "myapp/services/user_service.py",
            "content": '''"""User service with database queries."""
from django.db import connection
from django.contrib.auth.models import User


class UserService:
    """Service for user-related database operations."""

    def search_users(self, query: str):
        """Search users by username or email."""
        # SECURITY DRIFT: SQL Injection vulnerability!
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT * FROM auth_user WHERE username LIKE '%{query}%' OR email LIKE '%{query}%'"
            )
            return cursor.fetchall()

    def get_user_by_id(self, user_id):
        """Get user by ID."""
        # SECURITY DRIFT: Another SQL injection
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM auth_user WHERE id = {user_id}")
            return cursor.fetchone()

    def get_active_users(self, limit: int = 50):
        """Get active users."""
        # SECURITY DRIFT: SQL injection in limit
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM auth_user WHERE is_active = 1 LIMIT {limit}")
            return cursor.fetchall()'''
        }
    },
    "signal": {
        "gold": {
            "path": "myapp/signals/user_signals.py",
            "content": '''"""User-related Django signals."""
from django.db.models.signals import post_save, pre_delete
from django.dispatch import receiver
from django.contrib.auth.models import User
from django.core.cache import cache
import logging


logger = logging.getLogger(__name__)


@receiver(post_save, sender=User)
def user_saved_handler(sender, instance, created, **kwargs):
    """Handle user save events.

    - Invalidates user cache
    - Logs the event
    - Creates profile for new users
    """
    # Invalidate cache
    cache_key = f"user:{instance.pk}"
    cache.delete(cache_key)

    if created:
        logger.info("New user created: %s", instance.username)
        # Create user profile
        from myapp.models import UserProfile
        UserProfile.objects.get_or_create(user=instance)
    else:
        logger.debug("User updated: %s", instance.username)


@receiver(pre_delete, sender=User)
def user_pre_delete_handler(sender, instance, **kwargs):
    """Handle user deletion.

    Performs cleanup before user is deleted.
    """
    logger.info("User being deleted: %s", instance.username)
    # Clean up related data
    cache.delete(f"user:{instance.pk}")
    cache.delete(f"user_permissions:{instance.pk}")'''
        },
        "drift": {
            "path": "myapp/signals/user_signals.py",
            "content": '''"""User-related Django signals."""
from django.db.models.signals import post_save
from django.contrib.auth.models import User


def user_saved_handler(sender, instance, created, **kwargs):
    """Handle user save events."""
    # PATTERN DRIFT: Signal handler not using @receiver decorator
    if created:
        print(f"New user: {instance.username}")  # PATTERN DRIFT: Using print
        # LOGIC DRIFT: Circular import inside signal
        from myapp.models import UserProfile
        from myapp.services import send_welcome_email
        UserProfile.objects.create(user=instance)
        send_welcome_email(instance)  # LOGIC DRIFT: Sync email in signal


# PATTERN DRIFT: Manual signal connection (fragile)
post_save.connect(user_saved_handler, sender=User)


# LOGIC DRIFT: No pre_delete handler - orphaned data!'''
        }
    },
    "txn_pattern": {
        "gold": {
            "path": "myapp/services/order_service.py",
            "content": '''"""Order service with proper transaction handling."""
from django.db import transaction
from django.core.exceptions import ValidationError
from decimal import Decimal
from typing import List
import logging

from myapp.models import Order, OrderItem, Inventory


logger = logging.getLogger(__name__)


class OrderService:
    """Service for order operations with atomic transactions."""

    @transaction.atomic
    def create_order(self, user, items: List[dict]) -> Order:
        """Create order with items atomically.

        All database operations are wrapped in a transaction.
        If any step fails, everything is rolled back.
        """
        # Create order
        order = Order.objects.create(
            user=user,
            status='pending',
            total=Decimal('0')
        )

        total = Decimal('0')

        for item_data in items:
            # Lock inventory row to prevent race conditions
            inventory = Inventory.objects.select_for_update().get(
                product_id=item_data['product_id']
            )

            if inventory.quantity < item_data['quantity']:
                raise ValidationError(f"Insufficient stock for {item_data['product_id']}")

            # Decrease inventory
            inventory.quantity -= item_data['quantity']
            inventory.save()

            # Create order item
            item = OrderItem.objects.create(
                order=order,
                product_id=item_data['product_id'],
                quantity=item_data['quantity'],
                price=item_data['price']
            )

            total += item.price * item.quantity

        order.total = total
        order.save()

        logger.info("Order %s created for user %s", order.id, user.id)
        return order'''
        },
        "drift": {
            "path": "myapp/services/order_service.py",
            "content": '''"""Order service."""
from myapp.models import Order, OrderItem, Inventory


class OrderService:
    """Service for order operations."""

    def create_order(self, user, items):
        """Create order with items."""
        # LOGIC DRIFT: No transaction - partial failures possible!
        order = Order.objects.create(user=user, status='pending', total=0)

        total = 0

        for item_data in items:
            # LOGIC DRIFT: No locking - race conditions possible!
            inventory = Inventory.objects.get(product_id=item_data['product_id'])

            # LOGIC DRIFT: Check without lock
            if inventory.quantity < item_data['quantity']:
                # Order already created but items not - inconsistent state!
                return None

            inventory.quantity -= item_data['quantity']
            inventory.save()

            # If this fails, inventory already decreased!
            item = OrderItem.objects.create(
                order=order,
                product_id=item_data['product_id'],
                quantity=item_data['quantity'],
                price=item_data['price']
            )

            total += item.price * item.quantity

        order.total = total
        order.save()

        return order'''
        }
    },
    "url_stale": {
        "gold": {
            "path": "myapp/urls.py",
            "content": '''"""URL configuration using modern Django patterns."""
from django.urls import path, include
from myapp import views


app_name = 'myapp'

urlpatterns = [
    # Modern path() syntax with type converters
    path('', views.index, name='index'),
    path('users/', views.UserListView.as_view(), name='user-list'),
    path('users/<int:pk>/', views.UserDetailView.as_view(), name='user-detail'),
    path('users/<int:user_id>/posts/', views.user_posts, name='user-posts'),

    # Nested URL includes
    path('api/v1/', include('myapp.api.urls', namespace='api-v1')),

    # Slug-based URLs
    path('posts/<slug:slug>/', views.post_detail, name='post-detail'),

    # UUID URLs
    path('orders/<uuid:order_id>/', views.order_detail, name='order-detail'),
]'''
        },
        "drift": {
            "path": "myapp/urls.py",
            "content": '''"""URL configuration."""
from django.conf.urls import url  # STALE: Deprecated import
from myapp import views


# STALE DRIFT: Using deprecated url() instead of path()
urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^users/$', views.user_list, name='user-list'),
    url(r'^users/(?P<pk>[0-9]+)/$', views.user_detail, name='user-detail'),
    url(r'^users/(?P<user_id>[0-9]+)/posts/$', views.user_posts, name='user-posts'),

    # STALE: Regex patterns instead of type converters
    url(r'^posts/(?P<slug>[-\w]+)/$', views.post_detail, name='post-detail'),
    url(r'^orders/(?P<order_id>[a-f0-9-]+)/$', views.order_detail, name='order-detail'),
]'''
        }
    },
    "view_logic": {
        "gold": {
            "path": "myapp/views/auth_views.py",
            "content": '''"""Authentication views with proper logic."""
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_protect


@require_http_methods(["GET", "POST"])
@csrf_protect
def login_view(request):
    """Handle user login."""
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')

        if not username or not password:
            messages.error(request, 'Please provide both username and password.')
            return render(request, 'auth/login.html')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            if user.is_active:
                login(request, user)
                messages.success(request, 'Welcome back!')
                next_url = request.GET.get('next', 'dashboard')
                return redirect(next_url)
            else:
                messages.error(request, 'Your account is disabled.')
        else:
            messages.error(request, 'Invalid username or password.')

    return render(request, 'auth/login.html')


@login_required
def logout_view(request):
    """Handle user logout."""
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('login')'''
        },
        "drift": {
            "path": "myapp/views/auth_views.py",
            "content": '''"""Authentication views."""
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.contrib.auth.models import User


def login_view(request):
    """Handle user login."""
    # LOGIC DRIFT: No redirect for already authenticated users

    if request.method == 'POST':
        username = request.POST['username']  # LOGIC DRIFT: No error handling
        password = request.POST['password']

        # SECURITY DRIFT: No CSRF protection decorator

        # LOGIC DRIFT: Custom auth instead of Django's authenticate
        try:
            user = User.objects.get(username=username)
            if user.password == password:  # SECURITY DRIFT: Plain text comparison!
                request.session['user_id'] = user.id  # LOGIC DRIFT: Manual session
                return redirect('/')
        except User.DoesNotExist:
            pass

        return render(request, 'auth/login.html', {'error': 'Invalid'})

    return render(request, 'auth/login.html')


def logout_view(request):
    """Handle user logout."""
    # LOGIC DRIFT: No login_required check
    if 'user_id' in request.session:
        del request.session['user_id']
    return redirect('/')'''
        }
    }
}


# ============================================================================
# TANSTACK QUERY PATCHES (TypeScript)
# ============================================================================

TANSTACK_QUERY_PATCHES = {
    "hook_error": {
        "gold": {
            "path": "src/hooks/useUser.ts",
            "content": '''import { useQuery, UseQueryOptions } from '@tanstack/react-query';
import { fetchUser, User } from '../api/users';

interface UseUserOptions {
  userId: string;
  enabled?: boolean;
}

/**
 * Hook to fetch user data with proper error handling.
 */
export function useUser({ userId, enabled = true }: UseUserOptions) {
  return useQuery<User, Error>({
    queryKey: ['user', userId],
    queryFn: () => fetchUser(userId),
    enabled: enabled && !!userId,
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes (v5 renamed from cacheTime)
    retry: (failureCount, error) => {
      // Don't retry on 404s
      if (error.message.includes('404')) return false;
      return failureCount < 3;
    },
    meta: {
      errorMessage: 'Failed to load user profile',
    },
  });
}'''
        },
        "drift": {
            "path": "src/hooks/useUser.ts",
            "content": '''import { useQuery } from '@tanstack/react-query';
import { fetchUser } from '../api/users';

/**
 * Hook to fetch user data.
 */
export function useUser(userId: string) {
  // PATTERN DRIFT: No TypeScript generics
  // LOGIC DRIFT: No enabled check - fetches even without userId
  // STALE DRIFT: Using deprecated cacheTime instead of gcTime
  return useQuery({
    queryKey: ['user'],  // LOGIC DRIFT: Missing userId in key - causes stale data
    queryFn: () => fetchUser(userId),
    cacheTime: 1000 * 60 * 10, // STALE: Should be gcTime in v5
    // LOGIC DRIFT: No retry configuration
    // LOGIC DRIFT: No error handling meta
  });
}'''
        }
    },
    "query_key": {
        "gold": {
            "path": "src/hooks/useProducts.ts",
            "content": '''import { useQuery, useInfiniteQuery } from '@tanstack/react-query';
import { fetchProducts, ProductFilters, ProductsResponse } from '../api/products';

// Query key factory for consistent key generation
export const productKeys = {
  all: ['products'] as const,
  lists: () => [...productKeys.all, 'list'] as const,
  list: (filters: ProductFilters) => [...productKeys.lists(), filters] as const,
  details: () => [...productKeys.all, 'detail'] as const,
  detail: (id: string) => [...productKeys.details(), id] as const,
};

interface UseProductsOptions {
  filters: ProductFilters;
  enabled?: boolean;
}

/**
 * Hook to fetch paginated products with proper query key structure.
 */
export function useProducts({ filters, enabled = true }: UseProductsOptions) {
  return useQuery<ProductsResponse, Error>({
    queryKey: productKeys.list(filters),
    queryFn: () => fetchProducts(filters),
    enabled,
    staleTime: 30 * 1000, // 30 seconds
    placeholderData: (previousData) => previousData,
  });
}

/**
 * Hook for infinite scrolling products.
 */
export function useInfiniteProducts(filters: Omit<ProductFilters, 'page'>) {
  return useInfiniteQuery({
    queryKey: [...productKeys.lists(), 'infinite', filters],
    queryFn: ({ pageParam = 1 }) => fetchProducts({ ...filters, page: pageParam }),
    getNextPageParam: (lastPage) => lastPage.nextPage ?? undefined,
    initialPageParam: 1,
  });
}'''
        },
        "drift": {
            "path": "src/hooks/useProducts.ts",
            "content": '''import { useQuery } from '@tanstack/react-query';
import { fetchProducts } from '../api/products';

/**
 * Hook to fetch products.
 */
export function useProducts(category: string, page: number) {
  // PATTERN DRIFT: No query key factory
  // LOGIC DRIFT: Query key doesn't include all dependencies
  return useQuery({
    queryKey: ['products', category],  // Missing page - causes stale data!
    queryFn: () => fetchProducts({ category, page }),
    // STALE DRIFT: Using deprecated keepPreviousData
    keepPreviousData: true,  // Should use placeholderData in v5
  });
}

export function useInfiniteProducts(category: string) {
  // PATTERN DRIFT: Using regular useQuery for infinite scroll
  return useQuery({
    queryKey: ['products', 'infinite'],
    queryFn: () => fetchProducts({ category, page: 1 }),
    // LOGIC DRIFT: No pagination handling
  });
}'''
        }
    },
    "stale_time": {
        "gold": {
            "path": "src/hooks/useConfig.ts",
            "content": '''import { useQuery, useSuspenseQuery } from '@tanstack/react-query';
import { fetchAppConfig, AppConfig } from '../api/config';

/**
 * Hook for app configuration that rarely changes.
 * Uses appropriate staleTime and gcTime for static data.
 */
export function useAppConfig() {
  return useQuery<AppConfig, Error>({
    queryKey: ['config', 'app'],
    queryFn: fetchAppConfig,
    staleTime: Infinity, // Config never becomes stale during session
    gcTime: Infinity, // Keep in cache forever
    refetchOnMount: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  });
}

/**
 * Hook for user preferences with moderate cache time.
 */
export function useUserPreferences(userId: string) {
  return useQuery({
    queryKey: ['config', 'preferences', userId],
    queryFn: () => fetchUserPreferences(userId),
    staleTime: 5 * 60 * 1000, // 5 minutes - preferences change occasionally
    gcTime: 30 * 60 * 1000, // 30 minutes
    enabled: !!userId,
  });
}

/**
 * Suspense version for use with React Suspense.
 */
export function useAppConfigSuspense() {
  return useSuspenseQuery<AppConfig, Error>({
    queryKey: ['config', 'app'],
    queryFn: fetchAppConfig,
    staleTime: Infinity,
  });
}

async function fetchUserPreferences(userId: string) {
  const response = await fetch(\`/api/users/\${userId}/preferences\`);
  if (!response.ok) throw new Error('Failed to fetch preferences');
  return response.json();
}'''
        },
        "drift": {
            "path": "src/hooks/useConfig.ts",
            "content": '''import { useQuery } from '@tanstack/react-query';
import { fetchAppConfig } from '../api/config';

/**
 * Hook for app configuration.
 */
export function useAppConfig() {
  // LOGIC DRIFT: No staleTime - refetches constantly
  // LOGIC DRIFT: No gcTime - gets garbage collected quickly
  return useQuery({
    queryKey: ['config'],
    queryFn: fetchAppConfig,
    // Using defaults causes unnecessary refetches of static data
  });
}

export function useUserPreferences(userId: string) {
  return useQuery({
    queryKey: ['preferences'],  // LOGIC DRIFT: Missing userId in key
    queryFn: () => fetch(\`/api/users/\${userId}/preferences\`).then(r => r.json()),
    // LOGIC DRIFT: No enabled check - fetches without userId
    // LOGIC DRIFT: No error handling for fetch
  });
}'''
        }
    },
    "v5_options": {
        "gold": {
            "path": "src/hooks/useMutation.ts",
            "content": '''import { useMutation, useQueryClient } from '@tanstack/react-query';
import { createUser, updateUser, deleteUser, User, CreateUserInput } from '../api/users';

/**
 * Hook for creating users with optimistic updates.
 */
export function useCreateUser() {
  const queryClient = useQueryClient();

  return useMutation<User, Error, CreateUserInput>({
    mutationFn: createUser,
    onMutate: async (newUser) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: ['users'] });

      // Snapshot previous value
      const previousUsers = queryClient.getQueryData<User[]>(['users']);

      // Optimistically update
      queryClient.setQueryData<User[]>(['users'], (old) => [
        ...(old ?? []),
        { ...newUser, id: 'temp-id' } as User,
      ]);

      return { previousUsers };
    },
    onError: (err, newUser, context) => {
      // Rollback on error
      queryClient.setQueryData(['users'], context?.previousUsers);
    },
    onSettled: () => {
      // Always refetch after mutation
      queryClient.invalidateQueries({ queryKey: ['users'] });
    },
  });
}

/**
 * Hook for deleting users.
 */
export function useDeleteUser() {
  const queryClient = useQueryClient();

  return useMutation<void, Error, string>({
    mutationFn: deleteUser,
    onSuccess: (_, userId) => {
      // Remove from cache immediately
      queryClient.setQueryData<User[]>(['users'], (old) =>
        old?.filter((user) => user.id !== userId)
      );
      // Invalidate to ensure consistency
      queryClient.invalidateQueries({ queryKey: ['users'] });
    },
  });
}'''
        },
        "drift": {
            "path": "src/hooks/useMutation.ts",
            "content": '''import { useMutation, useQueryClient } from '@tanstack/react-query';
import { createUser, deleteUser } from '../api/users';

/**
 * Hook for creating users.
 */
export function useCreateUser() {
  const queryClient = useQueryClient();

  // STALE DRIFT: Using v4 options syntax
  return useMutation(createUser, {
    onSuccess: () => {
      // STALE DRIFT: Old invalidateQueries syntax
      queryClient.invalidateQueries('users');  // Should be { queryKey: ['users'] }
    },
  });
}

export function useDeleteUser() {
  const queryClient = useQueryClient();

  // STALE DRIFT: v4 syntax with function as first arg
  return useMutation(
    (userId: string) => deleteUser(userId),
    {
      // LOGIC DRIFT: No optimistic update
      // LOGIC DRIFT: No error handling
      onSuccess: () => {
        queryClient.invalidateQueries('users');
      },
    }
  );
}'''
        }
    }
}


# ============================================================================
# SHADCN-UI PATCHES (TypeScript/React)
# ============================================================================

SHADCN_PATCHES = {
    "button_variants": {
        "gold": {
            "path": "packages/ui/src/components/button.tsx",
            "content": '''import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
        outline: "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-md px-3",
        lg: "h-11 rounded-md px-8",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };'''
        },
        "drift": {
            "path": "packages/ui/src/components/button.tsx",
            "content": '''import * as React from "react";

// PATTERN DRIFT: Inline styles instead of CVA
// PATTERN DRIFT: No variant system

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary";
  size?: "small" | "large";
}

// PATTERN DRIFT: Not using forwardRef
function Button({ variant = "primary", size, className, ...props }: ButtonProps) {
  // PATTERN DRIFT: Inline conditional styles
  const styles: React.CSSProperties = {
    padding: size === "small" ? "4px 8px" : size === "large" ? "12px 24px" : "8px 16px",
    backgroundColor: variant === "primary" ? "blue" : "gray",
    color: "white",
    border: "none",
    borderRadius: "4px",
  };

  return (
    <button style={styles} className={className} {...props} />
  );
}

export { Button };'''
        }
    },
    "circ_dep": {
        "gold": {
            "path": "packages/ui/src/lib/utils.ts",
            "content": '''import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Utility function to merge Tailwind CSS classes.
 * Combines clsx for conditional classes with tailwind-merge for deduplication.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/**
 * Format a date for display.
 */
export function formatDate(date: Date | string): string {
  const d = typeof date === "string" ? new Date(date) : date;
  return d.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

/**
 * Debounce a function call.
 */
export function debounce<T extends (...args: unknown[]) => unknown>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout>;
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
}'''
        },
        "drift": {
            "path": "packages/ui/src/lib/utils.ts",
            "content": '''import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";
// ARCHITECTURE DRIFT: Circular dependency - utils importing from components
import { Button } from "../components/button";
import { formatButtonLabel } from "../components/button-utils";

export function cn(...inputs: any[]) {
  return twMerge(clsx(inputs));
}

// ARCHITECTURE DRIFT: UI logic in utils file
export function createStyledButton(label: string) {
  return { label: formatButtonLabel(label), Component: Button };
}

// PATTERN DRIFT: Any types instead of proper typing
export function formatDate(date: any): string {
  return new Date(date).toString();
}

// ARCHITECTURE DRIFT: Side effects in utils
console.log("Utils loaded");'''
        }
    },
    "export_standard": {
        "gold": {
            "path": "packages/ui/src/components/alert-banner.tsx",
            "content": '''import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../lib/utils";
import { AlertCircle, CheckCircle, Info, XCircle } from "lucide-react";

const alertBannerVariants = cva(
  "flex items-center gap-3 rounded-lg border p-4",
  {
    variants: {
      variant: {
        default: "bg-background text-foreground",
        info: "border-blue-200 bg-blue-50 text-blue-900 dark:border-blue-800 dark:bg-blue-950 dark:text-blue-100",
        success: "border-green-200 bg-green-50 text-green-900 dark:border-green-800 dark:bg-green-950 dark:text-green-100",
        warning: "border-yellow-200 bg-yellow-50 text-yellow-900 dark:border-yellow-800 dark:bg-yellow-950 dark:text-yellow-100",
        error: "border-red-200 bg-red-50 text-red-900 dark:border-red-800 dark:bg-red-950 dark:text-red-100",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

const iconMap = {
  default: Info,
  info: Info,
  success: CheckCircle,
  warning: AlertCircle,
  error: XCircle,
};

export interface AlertBannerProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof alertBannerVariants> {
  title?: string;
}

const AlertBanner = React.forwardRef<HTMLDivElement, AlertBannerProps>(
  ({ className, variant = "default", title, children, ...props }, ref) => {
    const Icon = iconMap[variant ?? "default"];

    return (
      <div
        ref={ref}
        role="alert"
        className={cn(alertBannerVariants({ variant }), className)}
        {...props}
      >
        <Icon className="h-5 w-5 flex-shrink-0" />
        <div className="flex-1">
          {title && <p className="font-medium">{title}</p>}
          {children && <p className="text-sm opacity-90">{children}</p>}
        </div>
      </div>
    );
  }
);
AlertBanner.displayName = "AlertBanner";

export { AlertBanner, alertBannerVariants };'''
        },
        "drift": {
            "path": "packages/ui/src/components/alert-banner.tsx",
            "content": '''import * as React from "react";

// PATTERN DRIFT: Default export instead of named export
// PATTERN DRIFT: No CVA, no variants
// PATTERN DRIFT: Inline styles

interface Props {
  type?: string;
  title?: string;
  children?: React.ReactNode;
}

// PATTERN DRIFT: No forwardRef
// PATTERN DRIFT: No displayName
export default function AlertBanner({ type = "info", title, children }: Props) {
  // PATTERN DRIFT: Inline color logic instead of variants
  const bgColor = type === "error" ? "red" : type === "success" ? "green" : "blue";

  return (
    <div style={{
      backgroundColor: bgColor,
      padding: "16px",
      borderRadius: "8px",
      color: "white"
    }}>
      {title && <strong>{title}</strong>}
      {children}
    </div>
  );
}'''
        }
    },
    "theme_logic": {
        "gold": {
            "path": "packages/ui/src/components/theme-provider.tsx",
            "content": '''import * as React from "react";

type Theme = "dark" | "light" | "system";

interface ThemeProviderProps {
  children: React.ReactNode;
  defaultTheme?: Theme;
  storageKey?: string;
}

interface ThemeProviderState {
  theme: Theme;
  setTheme: (theme: Theme) => void;
}

const ThemeProviderContext = React.createContext<ThemeProviderState | undefined>(
  undefined
);

export function ThemeProvider({
  children,
  defaultTheme = "system",
  storageKey = "ui-theme",
}: ThemeProviderProps) {
  const [theme, setThemeState] = React.useState<Theme>(() => {
    if (typeof window === "undefined") return defaultTheme;
    return (localStorage.getItem(storageKey) as Theme) || defaultTheme;
  });

  React.useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove("light", "dark");

    if (theme === "system") {
      const systemTheme = window.matchMedia("(prefers-color-scheme: dark)")
        .matches
        ? "dark"
        : "light";
      root.classList.add(systemTheme);
    } else {
      root.classList.add(theme);
    }
  }, [theme]);

  const setTheme = React.useCallback((newTheme: Theme) => {
    localStorage.setItem(storageKey, newTheme);
    setThemeState(newTheme);
  }, [storageKey]);

  const value = React.useMemo(
    () => ({ theme, setTheme }),
    [theme, setTheme]
  );

  return (
    <ThemeProviderContext.Provider value={value}>
      {children}
    </ThemeProviderContext.Provider>
  );
}

export function useTheme() {
  const context = React.useContext(ThemeProviderContext);
  if (context === undefined) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
}'''
        },
        "drift": {
            "path": "packages/ui/src/components/theme-provider.tsx",
            "content": '''import * as React from "react";

// LOGIC DRIFT: Module-level state instead of context
let currentTheme = "light";

// LOGIC DRIFT: Direct DOM manipulation
export function setTheme(theme: string) {
  currentTheme = theme;
  // LOGIC DRIFT: No SSR safety check
  document.documentElement.className = theme;
  // LOGIC DRIFT: Synchronous localStorage (can block)
  localStorage.setItem("theme", theme);
}

export function getTheme() {
  return currentTheme;
}

// PATTERN DRIFT: No context provider, just passes children through
export function ThemeProvider({ children }: { children: React.ReactNode }) {
  // LOGIC DRIFT: Effect runs on every render (no deps)
  React.useEffect(() => {
    const saved = localStorage.getItem("theme");
    if (saved) setTheme(saved);
  });

  return <>{children}</>;
}

// PATTERN DRIFT: Hook doesn't use context
export function useTheme() {
  // LOGIC DRIFT: Returns stale value, no reactivity
  return { theme: currentTheme, setTheme };
}'''
        }
    },
    "tooltip_security": {
        "gold": {
            "path": "packages/ui/src/components/tooltip.tsx",
            "content": '''import * as React from "react";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import { cn } from "../lib/utils";

const TooltipProvider = TooltipPrimitive.Provider;
const Tooltip = TooltipPrimitive.Root;
const TooltipTrigger = TooltipPrimitive.Trigger;

const TooltipContent = React.forwardRef<
  React.ElementRef<typeof TooltipPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Content>
>(({ className, sideOffset = 4, children, ...props }, ref) => (
  <TooltipPrimitive.Content
    ref={ref}
    sideOffset={sideOffset}
    className={cn(
      "z-50 overflow-hidden rounded-md border bg-popover px-3 py-1.5 text-sm text-popover-foreground shadow-md animate-in fade-in-0 zoom-in-95",
      className
    )}
    {...props}
  >
    {/* Content is safely rendered as text, not HTML */}
    {children}
  </TooltipPrimitive.Content>
));
TooltipContent.displayName = TooltipPrimitive.Content.displayName;

export { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider };'''
        },
        "drift": {
            "path": "packages/ui/src/components/tooltip.tsx",
            "content": '''import * as React from "react";

interface TooltipProps {
  content: string;
  children: React.ReactNode;
}

// PATTERN DRIFT: Custom implementation instead of Radix
// SECURITY DRIFT: Using dangerouslySetInnerHTML
export function Tooltip({ content, children }: TooltipProps) {
  const [show, setShow] = React.useState(false);

  return (
    <div
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
      style={{ position: "relative", display: "inline-block" }}
    >
      {children}
      {show && (
        <div
          style={{
            position: "absolute",
            bottom: "100%",
            left: "50%",
            transform: "translateX(-50%)",
            background: "black",
            color: "white",
            padding: "4px 8px",
            borderRadius: "4px",
            whiteSpace: "nowrap",
          }}
          // SECURITY DRIFT: XSS vulnerability!
          dangerouslySetInnerHTML={{ __html: content }}
        />
      )}
    </div>
  );
}'''
        }
    }
}


def write_patches(repo_name: str, patches: dict, task_prefix: str, repo_full: str):
    """Write patches and task files for a repository."""
    repo_dir = os.path.join(DATASETS_DIR, repo_name)
    patches_dir = os.path.join(repo_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)

    for task_name, patch_data in patches.items():
        # Write gold patch
        gold_content = create_new_file_patch(
            patch_data["gold"]["path"],
            patch_data["gold"]["content"]
        )
        gold_path = os.path.join(patches_dir, f"{task_name}_gold.patch")
        with open(gold_path, 'w') as f:
            f.write(gold_content)

        # Write drift patch
        drift_content = create_new_file_patch(
            patch_data["drift"]["path"],
            patch_data["drift"]["content"]
        )
        drift_path = os.path.join(patches_dir, f"{task_name}_drift.patch")
        with open(drift_path, 'w') as f:
            f.write(drift_content)

        print(f"  ✅ {task_name}: gold + drift patches")


def main():
    print("🔧 Regenerating all dataset patches...\n")

    print("📦 Lodash patches:")
    write_patches("lodash", LODASH_PATCHES, "lodash", "lodash/lodash")

    print("\n📦 Flask patches:")
    write_patches("flask", FLASK_PATCHES, "flask", "pallets/flask")

    print("\n📦 FastAPI patches:")
    write_patches("fastapi", FASTAPI_PATCHES, "fastapi", "tiangolo/fastapi")

    print("\n📦 Django patches:")
    write_patches("django", DJANGO_PATCHES, "django", "django/django")

    print("\n📦 TanStack Query patches:")
    write_patches("tanstack-query", TANSTACK_QUERY_PATCHES, "query", "TanStack/query")

    print("\n📦 shadcn-ui patches:")
    write_patches("shadcn-ui", SHADCN_PATCHES, "shadcn", "shadcn-ui/ui")

    print("\n✅ All patches regenerated!")
    print("\nNext: Run validation script to verify patches apply correctly.")


if __name__ == "__main__":
    main()
