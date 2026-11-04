"""Command line.

    python -m joracle_mini list
    python -m joracle_mini run prompt_003848 --arm olens
    python -m joracle_mini eval --arms transcript,olens --draws 1
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from joracle_mini.agent import investigate
from joracle_mini.data import incidents
from joracle_mini.judge import judge
from joracle_mini.llm import LLM


def cmd_list(_args) -> None:
    for pid, inc in incidents().items():
        print(f"{pid}  yes in {inc.k}/{inc.n}  {inc.behaviour[:100]}")


def cmd_run(args) -> None:
    inc = incidents()[args.incident]
    agent_llm, judge_llm = LLM(args.model), LLM(args.judge_model)
    print(f"{inc.prompt_id} ({args.arm}): {inc.behaviour}\n")
    result = investigate(inc, args.arm, agent_llm, verbose=True)
    verdict = judge(inc, result["answer"], judge_llm)
    print(f"\n=== answer ===\n{result['answer']}")
    print(f"\n=== judge: {verdict['score']}/10 ({'pass' if verdict['passed'] else 'fail'}) ===\n{verdict['reason']}")
    print(f"\nmodel calls {agent_llm.calls + judge_llm.calls}, new cost ${agent_llm.cost + judge_llm.cost:.3f}")


def cmd_eval(args) -> None:
    arms = args.arms.split(",")
    ids = args.ids.split(",") if args.ids else list(incidents())
    jobs = [(pid, arm, d) for pid in ids for arm in arms for d in range(args.draws)]

    def one(job):
        pid, arm, draw = job
        agent_llm, judge_llm = LLM(args.model), LLM(args.judge_model)
        if draw:  # independent samples per draw; draw 0 shares the cache with `run`
            agent_llm.salt = judge_llm.salt = f"{agent_llm.salt}draw{draw}"
        inc = incidents()[pid]
        result = investigate(inc, arm, agent_llm)
        verdict = judge(inc, result["answer"], judge_llm)
        return {"prompt_id": pid, "arm": arm, "draw": draw, **verdict, "answer": result["answer"],
                "hypothesis": result["hypothesis"], "steps": result["steps"],
                "cost": agent_llm.cost + judge_llm.cost}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, jobs))

    print(f"{'incident':16s}" + "".join(f"{arm:>14s}" for arm in arms))
    for pid in ids:
        cells = []
        for arm in arms:
            scores = [r["score"] for r in rows if r["prompt_id"] == pid and r["arm"] == arm]
            cells.append(f"{sum(scores) / len(scores):>14.1f}")
        print(f"{pid:16s}" + "".join(cells))
    print("-" * (16 + 14 * len(arms)))
    print(f"{'pass@1':16s}" + "".join(
        f"{100 * sum(r['passed'] for r in rows if r['arm'] == arm) / sum(r['arm'] == arm for r in rows):>13.0f}%" for arm in arms))
    print(f"\nnew cost ${sum(r['cost'] for r in rows):.2f}")
    out = Path("results") / f"eval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="joracle_mini")
    parser.add_argument("--model", default=None, help="investigator model (OpenRouter id)")
    parser.add_argument("--judge-model", default=None, help="judge model (OpenRouter id)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    p = sub.add_parser("run")
    p.add_argument("incident")
    p.add_argument("--arm", choices=["transcript", "olens"], default="olens")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("eval")
    p.add_argument("--arms", default="transcript,olens")
    p.add_argument("--draws", type=int, default=1)
    p.add_argument("--ids", default="")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=cmd_eval)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
