"""Best-effort accessors for the single-writer stream fence (#65991).

The fence itself lives on ``AIAgent`` (``_claim_stream_writer`` /
``_stream_writer_is_current`` in ``run_agent.py``), but the streaming code paths
that use it live in *other* modules — ``chat_completion_helpers`` (chat /
anthropic / bedrock) and ``codex_runtime`` (codex responses). Calling the fence
directly as ``agent._claim_stream_writer()`` from those modules makes them
hard-depend on the method being present on whatever object is passed in as
``agent``.

That coupling is a latent crash: a partially-updated checkout (the streaming
helper module newer than ``run_agent``), a hot-reloaded gateway, a duck-typed
agent, or a test double without the method turns an *additive* safety net into a
fatal ``AttributeError`` that aborts the whole turn. A cron job died exactly
this way with ``'AIAgent' object has no attribute '_claim_stream_writer'``.

The fence is only ever allowed to drop a *provably* superseded stream — never
the sole legitimate writer. So when the guard is unavailable (or raises), the
correct degradation is "no fence": keep streaming. These helpers make the
claim/check best-effort to guarantee that.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def stream_writer_generation(agent: Any) -> Optional[int]:
    """Snapshot the active writer generation for a later atomic claim."""
    ensure = getattr(agent, "_ensure_stream_writer_state", None)
    if callable(ensure):
        try:
            ensure()
        except Exception:
            logger.debug(
                "stream single-writer: state initialization failed",
                exc_info=True,
            )
            return None
    lock = getattr(agent, "_stream_writer_lock", None)
    if lock is None or not hasattr(agent, "_stream_writer_token"):
        return None
    try:
        with lock:
            return int(agent._stream_writer_token)
    except Exception:
        logger.debug(
            "stream single-writer: generation snapshot failed",
            exc_info=True,
        )
        return None


def claim_stream_writer(
    agent: Any,
    *,
    expected_generation: Optional[int] = None,
    is_valid: Optional[Callable[[], bool]] = None,
) -> Optional[int]:
    """Claim the delta sink for the calling stream attempt, best-effort.

    Returns the agent's monotonic writer token when the fence is available,
    ``0`` when an atomic ``expected_generation`` / ``is_valid`` guard rejected
    the claim, or ``None`` when the fence is unavailable.  The distinction is
    important for guarded callers: rejection proves the worker is stale, while
    an unavailable fence must preserve the historical unfenced degradation.
    """
    claim = getattr(agent, "_claim_stream_writer", None)
    if callable(claim):
        try:
            return int(
                claim(
                    expected_generation=expected_generation,
                    is_valid=is_valid,
                )
            )
        except TypeError:
            # Version-skewed/duck-typed agents expose the original no-argument
            # claim. A guarded caller cannot safely split its validity check
            # from that claim, because a newer writer may win between them.
            # Degrade without claiming instead; callback-level turn fences still
            # protect guarded paths such as Bedrock.
            if expected_generation is not None or is_valid is not None:
                return None
            try:
                return int(claim())
            except Exception:
                logger.debug(
                    "stream single-writer: legacy claim failed; proceeding unfenced",
                    exc_info=True,
                )
                return None
        except Exception:
            logger.debug(
                "stream single-writer: claim failed; proceeding unfenced",
                exc_info=True,
            )
            return None
    return None


def stream_writer_is_current(agent: Any, token: Optional[int]) -> bool:
    """True when ``token`` is still the active writer, best-effort.

    A falsy token (from a claim that no-oped) or an agent without the fence
    means we cannot prove supersession, so the stream is treated as current and
    never fenced. This preserves the single-writer invariant's one-way promise:
    only a demonstrably stale writer is ever stopped.
    """
    if not token:
        return True
    is_current = getattr(agent, "_stream_writer_is_current", None)
    if callable(is_current):
        try:
            return bool(is_current(token))
        except Exception:
            logger.debug(
                "stream single-writer: is_current check failed; treating as current",
                exc_info=True,
            )
    return True
