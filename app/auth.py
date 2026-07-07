"""API key generation and HMAC-based hashing for system user authentication."""

import hashlib
import hmac
import secrets


def generate_api_key() -> str:
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str, secret: str) -> str:
    return hmac.new(secret.encode(), api_key.encode(), hashlib.sha256).hexdigest()


def verify_api_key(api_key: str, stored_hash: str, secret: str) -> bool:
    return hmac.compare_digest(hash_api_key(api_key, secret), stored_hash)
