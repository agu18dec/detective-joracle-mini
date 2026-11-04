"""Score an explanation against the causes CHIVE established by experiment.

The judge sees the ground-truth causes (and the alternatives CHIVE ruled out),
and gives the answer 0-10 for how well it names the same mechanism as any one
established cause. An answer passes at 7 or more, as in the full study.
"""

import json
import re

from joracle_mini.data import Incident
from joracle_mini.llm import LLM

PASS = 7

PROMPT = """You grade an explanation of why a language model shows a behaviour. Below is the \
reference: the causes that counterfactual experiments established, with their evidence, and \
alternatives that were ruled out.

{reference}

Score the candidate explanation from 0 to 10:
10 = names the same mechanism as an established cause, specifically.
7-9 = the same mechanism, less specific or with minor extra claims.
4-6 = partly right (the right area of the prompt, the wrong mechanism, or too vague to test).
0-3 = a different or ruled-out cause, or no real explanation.

Reply with JSON only: {{"score": <0-10>, "reason": "<one sentence>"}}"""


def judge(incident: Incident, answer: str, llm: LLM) -> dict:
    reply = llm.chat(
        [
            {"role": "system", "content": PROMPT.format(reference=incident.judge_reference)},
            {"role": "user", "content": f"Candidate explanation:\n{answer}"},
        ],
        max_tokens=1500,
    )["content"]
    match = re.search(r"\{.*\}", reply, re.S)
    try:
        verdict = json.loads(match.group(0)) if match else {}
        score = int(verdict.get("score"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"score": 0, "passed": False, "reason": f"unparseable judge reply: {reply[:200]}"}
    return {"score": score, "passed": score >= PASS, "reason": str(verdict.get("reason", ""))}
