"""The investigator's tools over one incident.

The readout tools read precomputed OLens readouts: for every token of the
conversation and each of 11 layers (20, 24, ..., 60) of Qwen3.6-27B, four short
texts that a trained reader decoded from the activation at that point.

Text first: the readout tools are locked until the agent commits a hypothesis
from the transcript alone. In the full study this protocol was the change that
helped (readouts stopped distracting the agent from answers it would get right
anyway); it should overturn the hypothesis only on contrary evidence.
"""

import json
import re

from joracle_mini.data import Incident, readouts

MAX_HITS = 12
SNIPPET = 160


def _show(text: str) -> str:
    return text.replace("\n", "\\n")


class Toolbox:
    def __init__(self, incident: Incident, arm: str) -> None:
        self.incident = incident
        self.arm = arm  # "transcript" or "olens"
        self.bank = readouts(incident.prompt_id) if arm == "olens" else None
        self.hypothesis: str | None = None
        self.answer: str | None = None

    # -- schemas the model sees ------------------------------------------

    def schemas(self) -> list[dict]:
        def fn(name, description, properties, required):
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {"type": "object", "properties": properties, "required": required},
                },
            }

        submit = fn(
            "submit_answer",
            "Submit your final explanation of why the model shows the behaviour. Ends the investigation.",
            {"explanation": {"type": "string", "description": "The cause, as specific as you can make it."}},
            ["explanation"],
        )
        if self.arm == "transcript":
            return [submit]
        return [
            fn(
                "commit_hypothesis",
                "Commit your best explanation from the transcript alone. Required before the readout tools work.",
                {"hypothesis": {"type": "string"}},
                ["hypothesis"],
            ),
            fn(
                "search_readouts",
                "Regex search (case-insensitive) over all readouts. Returns matching positions, the token there, "
                "its role, the layer, and the matching text.",
                {
                    "pattern": {"type": "string"},
                    "layers": {"type": "array", "items": {"type": "integer"}, "description": "Optional subset of 20..60."},
                },
                ["pattern"],
            ),
            fn(
                "inspect_position",
                "Show the tokens around one position and that position's readouts at every layer.",
                {"position": {"type": "integer"}},
                ["position"],
            ),
            submit,
        ]

    # -- dispatch -----------------------------------------------------------

    def call(self, name: str, arguments: str) -> str:
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return f"Error: arguments are not valid JSON ({exc})."
        try:
            if name == "submit_answer":
                self.answer = str(args.get("explanation", "")).strip()
                return "Submitted."
            if name == "commit_hypothesis":
                self.hypothesis = str(args.get("hypothesis", "")).strip()
                return "Hypothesis recorded. The readout tools are unlocked."
            if name in ("search_readouts", "inspect_position"):
                if self.bank is None:
                    return "Error: this arm has no readouts."
                if self.hypothesis is None:
                    return "Locked: call commit_hypothesis with your transcript-only explanation first."
                if name == "search_readouts":
                    return self.search(str(args["pattern"]), args.get("layers"))
                return self.inspect(int(args["position"]))
            return f"Error: unknown tool {name!r}."
        except Exception as exc:  # report tool errors to the model instead of crashing
            return f"Error in {name}: {exc}"

    def search(self, pattern: str, layers: list[int] | None = None) -> str:
        regex = re.compile(pattern, re.I)
        wanted = {str(layer) for layer in layers} if layers else None
        hits, roles = [], {}
        for layer, cells in self.bank["readouts"].items():
            if wanted and layer not in wanted:
                continue
            for pos, samples in cells.items():
                for sample in samples:
                    match = regex.search(sample)
                    if match:
                        p = int(pos)
                        role = self.incident.roles[p] if p < len(self.incident.roles) else "?"
                        roles[role] = roles.get(role, 0) + 1
                        start = max(0, match.start() - 60)
                        hits.append((p, int(layer), role, _show(sample[start : start + SNIPPET])))
                        break  # one hit per cell
        if not hits:
            return f"No readout matches /{pattern}/."
        hits.sort()
        lines = [f"{len(hits)} cells match /{pattern}/ (by role: {roles}). First {min(len(hits), MAX_HITS)}:"]
        for p, layer, role, text in hits[:MAX_HITS]:
            lines.append(f"- pos {p} {self.incident.pieces[p]!r} ({role}) L{layer}: ...{text}")
        return "\n".join(lines)

    def inspect(self, position: int) -> str:
        n = len(self.incident.pieces)
        if not 0 <= position < n:
            return f"Error: position must be in 0..{n - 1}."
        around = range(max(0, position - 4), min(n, position + 5))
        context = "".join(f"[{self.incident.pieces[i]}]" if i == position else self.incident.pieces[i] for i in around)
        lines = [f"pos {position} ({self.incident.roles[position]}), context: {_show(context)}"]
        for layer in self.bank["layers"]:
            samples = self.bank["readouts"][str(layer)].get(str(position), [])
            for sample in samples[:2]:
                lines.append(f"L{layer}: {_show(sample[:220])}")
        return "\n".join(lines)
