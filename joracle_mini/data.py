"""The sample incidents and their readouts.

``data/`` holds 8 incidents from the held-out split of the CHIVE forensics set
(Hugging Face ``agu18dec/chive-forensics``): the conversation, the behaviour,
the ground-truth causes established by CHIVE's counterfactual experiments, the
token pieces of the conversation, and one precomputed OLens readout file per
incident (``data/readouts/<prompt_id>.json.gz``).
"""

import gzip
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Incident:
    prompt_id: str
    messages: list[dict]  # the conversation before the response
    completion: str  # the target model's response
    behaviour: str  # yes/no question that defines the behaviour
    k: int  # samples (of n) that show the behaviour
    n: int
    judge_reference: str  # ground-truth causes, as written for the judge
    pieces: list[str]  # token strings of the full conversation, response included
    roles: list[str]  # role of each token: control, system, user, assistant, ...


def _rows(name: str) -> dict[str, dict]:
    with open(DATA_DIR / f"{name}.jsonl", encoding="utf-8") as f:
        return {row["prompt_id"]: row for row in map(json.loads, f)}


def _as_list(value):
    return json.loads(value) if isinstance(value, str) else value


@cache
def incidents() -> dict[str, Incident]:
    inc, truth, tokens = _rows("incidents"), _rows("ground_truth"), _rows("tokens")
    out = {}
    for pid, row in sorted(inc.items()):
        tok = tokens[pid]
        out[pid] = Incident(
            prompt_id=pid,
            messages=row["messages"],
            completion=row["completion"],
            behaviour=row["behaviour"],
            k=row["k"],
            n=row["n"],
            judge_reference=truth[pid]["judge_reference"],
            pieces=_as_list(tok["pieces"]),
            roles=[role for _, role in _as_list(tok["turn_of"])],
        )
    return out


@cache
def readouts(prompt_id: str) -> dict:
    """``{"layers": [...], "k": int, "readouts": {layer: {pos: [samples]}}, "token_ids": [...]}``."""
    path = DATA_DIR / "readouts" / f"{prompt_id}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)
