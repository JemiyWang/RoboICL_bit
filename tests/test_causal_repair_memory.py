import json

import pytest

from roboicl.policy.causal_repair_memory import (
    BriefAttemptTrace,
    PersistentRepairMemory,
    SessionKey,
    format_memory_context,
)


def make_trace(session, attempt=0, transition_id="t0", phase="approach"):
    return BriefAttemptTrace(
        session=session,
        attempt_id=attempt,
        transition_id=transition_id,
        start_step=0,
        end_step=2,
        phase=phase,
        requested_action=[[0.01, 0, 0]],
        executed_action=[[0.01, 0, 0]],
        execution_receipt={"executed_steps": 1, "unexecuted_steps": 0},
        before_state={"left_ee_pose": [0, 0, 0]},
        after_state={"left_ee_pose": [0.01, 0, 0]},
    )


def test_session_scoped_lifecycle_and_idempotent_commit(tmp_path):
    path = tmp_path / "memory.jsonl"
    session = SessionKey("insert_key", 7, 2)
    memory = PersistentRepairMemory(path, capacity=4, top_k=2)
    memory.begin_session(session)
    memory.begin_attempt(0)
    trace = make_trace(session)

    assert memory.append_transition(trace) is True
    assert memory.append_transition(trace) is False
    assert memory.complete_attempt(
        outcome="failure",
        failure_type="position_error",
        failure_evidence=["target not reached"],
        repair={"changed_factor": "z_height", "delta": -0.008},
    )["outcome"] == "failure"

    restored = PersistentRepairMemory(path, capacity=4, top_k=2)
    restored.begin_session(session)
    restored.begin_attempt(1)
    retrieved = restored.retrieve(phase="approach", failure_type="position_error")
    assert len(retrieved) == 1
    assert retrieved[0].outcome == "failure"
    assert retrieved[0].failure_type == "position_error"
    assert retrieved[0].diagnosis_source == "trial_end"


    other = SessionKey("insert_key", 7, 3)
    restored.begin_session(other)
    assert restored.retrieve() == []


def test_capacity_is_session_local_and_order_is_deterministic():
    session = SessionKey("deposit_coin", 0, 0)
    memory = PersistentRepairMemory(capacity=2, top_k=4)
    memory.begin_session(session)
    for attempt in range(3):
        memory.begin_attempt(attempt)
        memory.append_transition(make_trace(session, attempt, f"t{attempt}"))
    assert [item.transition_id for item in memory.entries] == ["t1", "t2"]
    memory.begin_attempt(3)
    assert [item.transition_id for item in memory.retrieve()] == ["t2", "t1"]


def test_rejects_conflicting_transition_and_invalid_json(tmp_path):
    session = SessionKey("general_pickup", 1, 0)
    memory = PersistentRepairMemory(tmp_path / "memory.jsonl")
    memory.begin_session(session)
    memory.begin_attempt(0)
    memory.append_transition(make_trace(session))
    with pytest.raises(ValueError, match="different content"):
        memory.append_transition(make_trace(session, transition_id="t0", phase="grasp"))

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"record_type": "unknown"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        PersistentRepairMemory(bad)


def test_prompt_formatter_marks_unverified_repairs():
    session = SessionKey("general_pickup", 0, 0)
    trace = make_trace(session)
    text = format_memory_context([trace], max_chars=10000)
    assert "repair_status=unverified" in text
    assert "observed_receipt" in text


def test_retrieval_excludes_current_attempt():
    session = SessionKey("general_pickup", 0, 0)
    memory = PersistentRepairMemory()
    memory.begin_session(session)
    memory.begin_attempt(0)
    memory.append_transition(make_trace(session, 0, "old"))
    memory.begin_attempt(1)
    memory.append_transition(make_trace(session, 1, "current"))
    assert [item.transition_id for item in memory.retrieve()] == ["old"]
