"""The investigator: a tool-calling loop over one incident."""

import json

from joracle_mini.data import Incident
from joracle_mini.llm import LLM
from joracle_mini.tools import Toolbox

MAX_STEPS = 16

SYSTEM = """You investigate why a language model (Qwen3.6-27B) behaves a certain way. You get one \
real conversation with it, the response it gave, and a yes/no question that defines a \
behaviour. In {k} of {n} samples of this response, the answer to that question was yes.

Your job: explain what in the conversation causes the behaviour. A good explanation names the \
specific feature of the prompt (a word, an instruction, a missing detail, the framing) and the \
mechanism, specifically enough that an experiment that edits that feature would change the rate.
{tools}
When you are done, call submit_answer with your explanation."""

OLENS_TOOLS = """
You also have readouts of the model's internal activations. For every token and 11 layers \
(20 to 60), a trained reader (OLens) wrote four short texts describing what the activation \
represents. They are noisy: trust patterns that repeat across layers and samples, not single \
lines. First commit a hypothesis from the transcript alone (commit_hypothesis); then use \
search_readouts and inspect_position to test it. Change your hypothesis only if the readouts \
contradict it. Some causes are absences (a missing instruction, a detail that is not there), \
and readouts cannot show an absence."""


def render_incident(incident: Incident) -> str:
    turns = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in incident.messages)
    return (
        f"Behaviour question: {incident.behaviour}\n"
        f"Rate: yes in {incident.k} of {incident.n} samples.\n\n"
        f"=== Conversation ===\n{turns}\n\n=== The model's response (one sample) ===\n{incident.completion}"
    )


def investigate(incident: Incident, arm: str, llm: LLM, verbose: bool = False) -> dict:
    """Run one episode. Returns {"answer", "hypothesis", "steps", "trace"}."""
    box = Toolbox(incident, arm)
    system = SYSTEM.format(k=incident.k, n=incident.n, tools=OLENS_TOOLS if arm == "olens" else "")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": render_incident(incident)}]
    trace = []
    for step in range(MAX_STEPS):
        reply = llm.chat(messages, tools=box.schemas())
        calls = reply["tool_calls"]
        messages.append(
            {
                "role": "assistant",
                "content": reply["content"],
                **(
                    {"tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]}
                    if calls
                    else {}
                ),
            }
        )
        if not calls:
            # A reply without a tool call: take it as the answer if nothing was submitted.
            trace.append({"step": step, "text": reply["content"]})
            if verbose and reply["content"]:
                print(f"\n[{step}] {reply['content'][:600]}")
            box.answer = box.answer or reply["content"].strip()
            break
        for c in calls:
            result = box.call(c["name"], c["arguments"])
            trace.append({"step": step, "tool": c["name"], "arguments": c["arguments"], "result": result})
            if verbose:
                print(f"\n[{step}] {c['name']}({c['arguments'][:200]})\n{result[:700]}")
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
        if box.answer is not None:
            break
    else:
        box.answer = box.answer or "(no answer submitted within the step limit)"
    return {"answer": box.answer, "hypothesis": box.hypothesis, "steps": len(trace), "trace": trace}


def dump(result: dict) -> str:
    return json.dumps(result, indent=2, ensure_ascii=False)
