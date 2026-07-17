"""
contracts/timestamp.py
======================
Canonical scenario-relative integer timestamp definitions and helpers.
"""
from __future__ import annotations

# Type alias
TimestampMs = int


def compute_age(current_ms: int, message_ms: int) -> int:
    """Compute elapsed age in milliseconds from message_ms to current_ms.

    Raises ValueError if message_ms is from the future (message_ms > current_ms).
    """
    c_int = int(current_ms)
    m_int = int(message_ms)
    if m_int > c_int:
        raise ValueError(
            f"Message timestamp ({m_int}) exceeds current scenario time ({c_int})."
        )
    return c_int - m_int


def is_fresh(current_ms: int, message_ms: int, max_age_ms: float) -> bool:
    """Return True if message_ms is fresh relative to current_ms within max_age_ms."""
    try:
        age = compute_age(current_ms, message_ms)
        return age <= int(max_age_ms)
    except ValueError:
        # A message from the future fails freshness checks
        return False
