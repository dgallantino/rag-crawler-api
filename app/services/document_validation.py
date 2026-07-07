"""Fast validation for uploaded markdown documents."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str | None = None


def validate_markdown_upload(filename: str, content: bytes) -> ValidationResult:
    if not filename.lower().endswith(".md"):
        return ValidationResult(valid=False, reason="Only .md files are accepted")

    if b"\x00" in content:
        return ValidationResult(valid=False, reason="File contains binary content")

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return ValidationResult(valid=False, reason="File must be valid UTF-8 text")

    if not text.strip():
        return ValidationResult(valid=False, reason="File content is empty")

    return ValidationResult(valid=True)
