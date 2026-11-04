"""Offline tests: no network, no API key. A fake LLM replays scripted replies."""

import json

from joracle_mini.agent import investigate
from joracle_mini.data import incidents, readouts
from joracle_mini.judge import judge
from joracle_mini.tools import Toolbox

PID = "prompt_003848"


class FakeLLM:
    """Replays a list of replies; each is text or a (tool name, args) pair."""

    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def chat(self, messages, tools=None, max_tokens=4000):
        self.seen.append((messages, tools))
        item = self.script.pop(0)
        if isinstance(item, tuple):
            name, args = item
            return {"content": "", "tool_calls": [{"id": f"c{len(self.seen)}", "name": name, "arguments": json.dumps(args)}]}
        return {"content": item, "tool_calls": []}


def test_sample_data_loads_and_matches_readouts():
    inc = incidents()
    assert len(inc) == 8
    for pid, incident in inc.items():
        bank = readouts(pid)
        assert len(bank["token_ids"]) == len(incident.pieces) == len(incident.roles)
        assert bank["layers"] == list(range(20, 61, 4))


def test_readout_tools_are_locked_until_a_hypothesis():
    box = Toolbox(incidents()[PID], "olens")
    assert box.call("search_readouts", '{"pattern": "role"}').startswith("Locked")
    box.call("commit_hypothesis", '{"hypothesis": "narrow role"}')
    assert "cells match" in box.call("search_readouts", '{"pattern": "role"}')


def test_inspect_shows_every_layer():
    box = Toolbox(incidents()[PID], "olens")
    box.call("commit_hypothesis", '{"hypothesis": "x"}')
    text = box.call("inspect_position", '{"position": 5}')
    assert all(f"L{layer}:" in text for layer in range(20, 61, 4))


def test_tool_errors_come_back_as_text():
    box = Toolbox(incidents()[PID], "olens")
    box.call("commit_hypothesis", '{"hypothesis": "x"}')
    assert box.call("search_readouts", '{"pattern": "("}').startswith("Error")
    assert box.call("inspect_position", '{"position": 99999}').startswith("Error")
    assert box.call("no_such_tool", "{}").startswith("Error")


def test_transcript_arm_only_has_submit():
    box = Toolbox(incidents()[PID], "transcript")
    assert [t["function"]["name"] for t in box.schemas()] == ["submit_answer"]


def test_episode_runs_to_submission():
    llm = FakeLLM([
        ("commit_hypothesis", {"hypothesis": "narrow role"}),
        ("search_readouts", {"pattern": "medical"}),
        ("submit_answer", {"explanation": "The narrow system role is read as a whitelist."}),
    ])
    result = investigate(incidents()[PID], "olens", llm)
    assert result["answer"] == "The narrow system role is read as a whitelist."
    assert result["hypothesis"] == "narrow role"
    assert len(llm.seen) == 3


def test_judge_parses_scores():
    inc = incidents()[PID]
    assert judge(inc, "x", FakeLLM(['{"score": 8, "reason": "same mechanism"}']))["passed"] is True
    assert judge(inc, "x", FakeLLM(['```json\n{"score": 3, "reason": "other"}\n```']))["passed"] is False
    assert judge(inc, "x", FakeLLM(["no json here"]))["score"] == 0
