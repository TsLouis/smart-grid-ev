from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from ..agents import HumanLLMAgent, LLMAdvisorAgent


def _sample_observation(advice: str | None = None) -> dict:
    observation = {
        "runtime_context": {
            "scenario": "jun30",
            "time_h": 8.0,
            "purpose": "review specialist proposals before visualization or dispatch",
        },
        "specialist_proposals": [
            {"agent_name": "charge_need_agent", "proposal_type": "charge_need", "payload": {"should_charge": True}},
            {"agent_name": "station_decision_agent", "proposal_type": "station_decision", "payload": {"best_station": 6}},
        ],
    }
    if advice:
        observation["summary"] = advice
    return observation


def main() -> int:
    parser = argparse.ArgumentParser(description="Check standalone LLM/human advisor agent wiring.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--human", action="store_true", help="Ask stdin for a human response that plays the LLM role")
    parser.add_argument("--advice", default="", help="Non-interactive advice text for the LLM proposal")
    args = parser.parse_args()

    observation = _sample_observation(args.advice or None)
    if args.human:
        print("Runtime context and specialist proposals are ready for LLM-agent review.")
        print(json.dumps(observation, ensure_ascii=False, indent=2))
        advice = input("Human-as-LLM advice: ").strip()
        agent = HumanLLMAgent(advice=advice or "No advice was entered.")
    else:
        agent = LLMAdvisorAgent()

    proposal = agent.propose(observation)
    if args.json:
        print(json.dumps(asdict(proposal), ensure_ascii=False, indent=2))
    else:
        print(f"{proposal.agent_name}: {proposal.proposal_type} confidence={proposal.confidence}")
        print(f"  {proposal.payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
