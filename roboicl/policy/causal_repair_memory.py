"""Small, auditable cross-attempt memory for RoboICL.

The first implementation deliberately stays independent of numpy, torch,
embeddings, and the simulator.  It stores completed execution evidence in an
append-only JSONL file and performs deterministic, session-scoped retrieval.
The policy can therefore use it in ``shadow`` mode before any prompt-facing
integration is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Iterable


SCHEMA_VERSION = "roboicl.causal_repair_memory.v1"


def _plain(value: Any) -> Any:
    """Convert common array/scalar values to strict JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _plain(tolist())
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


def _canonical(value: Any) -> str:
    return json.dumps(_plain(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class SessionKey:
    """Identity of one task instance whose attempts may share memory."""

    task: str
    seed: int
    layout: int
    variant: str = "standard"
    protocol_version: str = "v1"

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("session task must be a non-empty string")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("session seed must be a nonnegative integer")
        if type(self.layout) is not int or self.layout < 0:
            raise ValueError("session layout must be a nonnegative integer")
        for name in ("variant", "protocol_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"session {name} must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "seed": self.seed,
            "layout": self.layout,
            "variant": self.variant,
            "protocol_version": self.protocol_version,
        }

    @property
    def stable_id(self) -> str:
        digest = hashlib.sha256(_canonical(self.to_dict()).encode("utf-8")).hexdigest()
        return digest[:24]


@dataclass(frozen=True)
class BriefAttemptTrace:
    """One executed action-effect transition suitable for long-term memory."""

    session: SessionKey
    attempt_id: int
    transition_id: str
    start_step: int
    end_step: int
    phase: str = "unknown"
    requested_action: Any = None
    executed_action: Any = None
    execution_receipt: dict[str, Any] = field(default_factory=dict)
    before_state: dict[str, Any] = field(default_factory=dict)
    after_state: dict[str, Any] = field(default_factory=dict)
    outcome: str = "unknown"
    failure_type: str = "unknown"
    failure_evidence: tuple[str, ...] = ()
    repair: dict[str, Any] | None = None
    diagnosis_source: str = "execution"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.attempt_id) is not int or self.attempt_id < 0:
            raise ValueError("attempt_id must be a nonnegative integer")
        if not isinstance(self.transition_id, str) or not self.transition_id.strip():
            raise ValueError("transition_id must be a non-empty string")
        if type(self.start_step) is not int or type(self.end_step) is not int:
            raise ValueError("transition steps must be integers")
        if self.start_step < 0 or self.end_step < self.start_step:
            raise ValueError("transition steps must be ordered and nonnegative")
        for name in ("phase", "outcome", "failure_type", "diagnosis_source"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.failure_evidence, (tuple, list)):
            raise ValueError("failure_evidence must be a sequence of strings")
        if any(not isinstance(item, str) for item in self.failure_evidence):
            raise ValueError("failure_evidence entries must be strings")
        if not isinstance(self.execution_receipt, dict):
            raise ValueError("execution_receipt must be a mapping")
        if not isinstance(self.before_state, dict) or not isinstance(self.after_state, dict):
            raise ValueError("before_state and after_state must be mappings")
        if self.repair is not None and not isinstance(self.repair, dict):
            raise ValueError("repair must be a mapping or None")
        # Fail before a record reaches disk.  This also normalises numpy-like
        # scalars without taking a numpy dependency in the memory module.
        _canonical(self.to_dict())

    @property
    def key(self) -> tuple[str, str]:
        return self.session.stable_id, self.transition_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "session": self.session.to_dict(),
            "attempt_id": self.attempt_id,
            "transition_id": self.transition_id,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "phase": self.phase,
            "requested_action": _plain(self.requested_action),
            "executed_action": _plain(self.executed_action),
            "execution_receipt": _plain(self.execution_receipt),
            "before_state": _plain(self.before_state),
            "after_state": _plain(self.after_state),
            "outcome": self.outcome,
            "failure_type": self.failure_type,
            "failure_evidence": list(self.failure_evidence),
            "repair": _plain(self.repair),
            "diagnosis_source": self.diagnosis_source,
            "metadata": _plain(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BriefAttemptTrace":
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported causal memory schema version")
        session = SessionKey(**dict(value["session"]))
        return cls(
            session=session,
            attempt_id=int(value["attempt_id"]),
            transition_id=str(value["transition_id"]),
            start_step=int(value["start_step"]),
            end_step=int(value["end_step"]),
            phase=str(value.get("phase", "unknown")),
            requested_action=value.get("requested_action"),
            executed_action=value.get("executed_action"),
            execution_receipt=dict(value.get("execution_receipt", {})),
            before_state=dict(value.get("before_state", {})),
            after_state=dict(value.get("after_state", {})),
            outcome=str(value.get("outcome", "unknown")),
            failure_type=str(value.get("failure_type", "unknown")),
            failure_evidence=tuple(value.get("failure_evidence", ())),
            repair=dict(value["repair"]) if value.get("repair") is not None else None,
            diagnosis_source=str(value.get("diagnosis_source", "execution")),
            metadata=dict(value.get("metadata", {})),
        )


class PersistentRepairMemory:
    """Bounded JSONL memory with explicit session and attempt lifecycles."""

    def __init__(self, path: str | Path | None = None, *, capacity: int = 128,
                 top_k: int = 4) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        self.path = Path(path) if path is not None else None
        self.capacity = capacity
        self.top_k = top_k
        self._entries: list[BriefAttemptTrace] = []
        self._attempt_outcomes: dict[tuple[str, int], dict[str, Any]] = {}
        self._session: SessionKey | None = None
        self._attempt_id: int | None = None
        self._lock = threading.RLock()
        if self.path is not None and self.path.exists():
            self._load()

    @property
    def session(self) -> SessionKey | None:
        return self._session

    @property
    def attempt_id(self) -> int | None:
        return self._attempt_id

    @property
    def entries(self) -> tuple[BriefAttemptTrace, ...]:
        return tuple(self._entries)

    def begin_session(self, session: SessionKey, *, resume: bool = True) -> None:
        if not isinstance(session, SessionKey):
            raise TypeError("session must be a SessionKey")
        with self._lock:
            if self._session is not None and self._session != session:
                self._attempt_id = None
            self._session = session
            if not resume:
                self.clear_session(session)
                self._attempt_id = None

    def begin_attempt(self, attempt_id: int) -> None:
        if type(attempt_id) is not int or attempt_id < 0:
            raise ValueError("attempt_id must be a nonnegative integer")
        with self._lock:
            if self._session is None:
                raise RuntimeError("begin_session() is required before begin_attempt()")
            if self._attempt_id is not None and attempt_id < self._attempt_id:
                raise ValueError("attempt_id cannot move backwards within a session")
            self._attempt_id = attempt_id

    def append_transition(self, trace: BriefAttemptTrace) -> bool:
        """Append one transition; return False for an identical retry."""
        if not isinstance(trace, BriefAttemptTrace):
            raise TypeError("trace must be a BriefAttemptTrace")
        with self._lock:
            if self._session is None or self._session != trace.session:
                raise ValueError("transition session does not match active session")
            if self._attempt_id is None:
                raise RuntimeError("begin_attempt() is required before append_transition()")
            if trace.attempt_id != self._attempt_id:
                raise ValueError("transition attempt does not match active attempt")
            for existing in self._entries:
                if existing.key != trace.key:
                    continue
                if _canonical(existing.to_dict()) != _canonical(trace.to_dict()):
                    raise ValueError("transition id already exists with different content")
                return False
            self._entries.append(trace)
            self._enforce_capacity(trace.session)
            self._append_line({"record_type": "transition", "entry": trace.to_dict()})
            return True

    def complete_attempt(self, *, outcome: str, failure_type: str = "unknown",
                         failure_evidence: Iterable[str] = (),
                         repair: dict[str, Any] | None = None) -> dict[str, Any]:
        if not isinstance(outcome, str) or not outcome.strip():
            raise ValueError("outcome must be a non-empty string")
        if not isinstance(failure_type, str) or not failure_type.strip():
            raise ValueError("failure_type must be a non-empty string")
        evidence = tuple(failure_evidence)
        if any(not isinstance(item, str) for item in evidence):
            raise ValueError("failure_evidence entries must be strings")
        with self._lock:
            if self._session is None or self._attempt_id is None:
                raise RuntimeError("begin_session() and begin_attempt() are required")
            summary = {
                "outcome": outcome,
                "failure_type": failure_type,
                "failure_evidence": list(evidence),
                "repair": _plain(repair),
            }
            key = (self._session.stable_id, self._attempt_id)
            previous = self._attempt_outcomes.get(key)
            if previous is not None and _canonical(previous) != _canonical(summary):
                raise ValueError("attempt already completed with different result")
            if previous is None:
                self._attempt_outcomes[key] = summary
                self._append_line({
                    "record_type": "attempt_end",
                    "schema_version": SCHEMA_VERSION,
                    "session": self._session.to_dict(),
                    "attempt_id": self._attempt_id,
                    **summary,
                })
            return dict(summary)

    def retrieve(self, *, phase: str | None = None,
                 failure_type: str | None = None,
                 top_k: int | None = None) -> list[BriefAttemptTrace]:
        """Return deterministic, session-scoped evidence from past attempts."""
        with self._lock:
            if self._session is None:
                raise RuntimeError("begin_session() is required before retrieve()")
            limit = self.top_k if top_k is None else top_k
            if type(limit) is not int or limit < 1:
                raise ValueError("top_k must be a positive integer")
            # Long-term memory is cross-attempt evidence.  Current-attempt
            # transitions remain in RoboICL's LIVE history and must not be
            # re-injected as if they were prior experience.
            candidates = [entry for entry in self._entries
                          if entry.session == self._session
                          and (self._attempt_id is None or entry.attempt_id < self._attempt_id)]
            ranked = []
            for index, entry in enumerate(candidates):
                score = 0
                if phase and entry.phase == phase:
                    score += 8
                if failure_type and entry.failure_type == failure_type:
                    score += 5
                outcome = self._attempt_outcomes.get((entry.session.stable_id, entry.attempt_id), {})
                if outcome.get("outcome") == "success":
                    score += 3
                if outcome.get("failure_type") and entry.failure_type == outcome["failure_type"]:
                    score += 1
                # Later entries are preferred only after semantic matches.
                ranked.append((score, entry.attempt_id, entry.end_step, index, entry, outcome))
            ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3]))
            result = []
            for _score, _attempt, _step, _index, entry, outcome in ranked[:limit]:
                if outcome:
                    result.append(replace(
                        entry,
                        outcome=str(outcome["outcome"]),
                        failure_type=str(outcome["failure_type"]),
                        failure_evidence=tuple(outcome["failure_evidence"]),
                        repair=outcome.get("repair"),
                        diagnosis_source=("trial_end" if outcome["outcome"] != "unknown"
                                          else "terminal_boundary"),
                    ))
                else:
                    result.append(entry)
            return result

    def clear_session(self, session: SessionKey | None = None) -> int:
        """Remove one session's entries and append a tombstone to JSONL."""
        with self._lock:
            target = session or self._session
            if target is None:
                return 0
            old = len(self._entries)
            self._entries = [entry for entry in self._entries if entry.session != target]
            for key in list(self._attempt_outcomes):
                if key[0] == target.stable_id:
                    del self._attempt_outcomes[key]
            removed = old - len(self._entries)
            if removed and self.path is not None:
                self._append_line({"record_type": "clear_session",
                                   "schema_version": SCHEMA_VERSION,
                                   "session": target.to_dict()})
            return removed

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": SCHEMA_VERSION,
                "active_session": self._session.to_dict() if self._session else None,
                "active_attempt": self._attempt_id,
                "entries": len(self._entries),
                "session_entries": (sum(entry.session == self._session for entry in self._entries)
                                    if self._session else 0),
                "capacity": self.capacity,
                "top_k": self.top_k,
            }

    def _enforce_capacity(self, session: SessionKey) -> None:
        matching = [entry for entry in self._entries if entry.session == session]
        if len(matching) <= self.capacity:
            return
        remove = len(matching) - self.capacity
        removed = set(id(entry) for entry in matching[:remove])
        self._entries = [entry for entry in self._entries if id(entry) not in removed]

    def _append_line(self, value: dict[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = _canonical(value) + "\n"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()

    def _load(self) -> None:
        assert self.path is not None
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    kind = record.get("record_type")
                    if kind == "transition":
                        trace = BriefAttemptTrace.from_dict(record["entry"])
                        # A repeated append can occur after a process is
                        # interrupted between flushes.  Keep recovery idempotent.
                        existing = next((item for item in self._entries if item.key == trace.key), None)
                        if existing is None:
                            self._entries.append(trace)
                        elif _canonical(existing.to_dict()) != _canonical(trace.to_dict()):
                            raise ValueError("conflicting transition records")
                    elif kind == "attempt_end":
                        session = SessionKey(**dict(record["session"]))
                        key = (session.stable_id, int(record["attempt_id"]))
                        self._attempt_outcomes[key] = {
                            "outcome": str(record["outcome"]),
                            "failure_type": str(record.get("failure_type", "unknown")),
                            "failure_evidence": list(record.get("failure_evidence", [])),
                            "repair": record.get("repair"),
                        }
                    elif kind == "clear_session":
                        session = SessionKey(**dict(record["session"]))
                        self._entries = [item for item in self._entries if item.session != session]
                        for key in list(self._attempt_outcomes):
                            if key[0] == session.stable_id:
                                del self._attempt_outcomes[key]
                    else:
                        raise ValueError(f"unknown record_type {kind!r}")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid causal memory record at line {line_number}: {exc}") from exc
        # Capacity is a logical bound.  Older JSONL records remain as an audit
        # trail, while the recovered working set obeys the current bound.
        sessions = {entry.session for entry in self._entries}
        for session in sessions:
            self._enforce_capacity(session)


def format_memory_context(entries: Iterable[BriefAttemptTrace], *, max_chars: int = 4000) -> str:
    """Render retrieved records without implying that hypotheses are facts."""
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    blocks = []
    for entry in entries:
        outcome = entry.outcome
        block = (
            f"attempt={entry.attempt_id} phase={entry.phase} outcome={outcome} "
            f"failure_type={entry.failure_type}\n"
            f"observed_receipt={_canonical(entry.execution_receipt)}\n"
            f"observed_after_state={_canonical(entry.after_state)}\n"
            f"failure_evidence={_canonical(list(entry.failure_evidence))}\n"
            f"repair_status={'verified' if entry.diagnosis_source == 'trial_end' else 'unverified'} "
            f"repair={_canonical(entry.repair)}"
        )
        blocks.append(block)
    result = "\n\n".join(blocks)
    if len(result) <= max_chars:
        return result
    return result[:max_chars].rstrip() + "\n[causal memory truncated]"


__all__ = [
    "SCHEMA_VERSION",
    "SessionKey",
    "BriefAttemptTrace",
    "PersistentRepairMemory",
    "format_memory_context",
]
