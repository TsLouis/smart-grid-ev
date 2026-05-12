# LLM Agent Design

## Position

LLM capability is modeled as a normal intelligent agent, parallel to the existing specialist agents. It is not embedded inside charge-need, station-decision, grid-friendly, queue, pricing, or forecasting agents.

The current implementation provides two swappable forms:

- `LLMAdvisorAgent`: a placeholder/API-ready advisor that emits the same structured proposal contract.
- `HumanLLMAgent`: a human-in-the-loop stand-in so the full interaction path can be tested before an LLM provider is attached.

## Contract

Both forms emit:

- `proposal_type`: `llm_advice`
- `payload.source`: `placeholder` or `human`
- `payload.summary`: high-level advice
- `payload.strategy_review`: optional review of expert proposals
- `payload.explanation`: optional user-facing explanation

The proposal is advisory only. It does not mutate digital twin state, runtime bindings, topology, event queues, load curves, or visualization state.

## Runtime Wiring

The default manifest adds a `runtime` T-layer manifest with an `llm_advisor` slot. This keeps LLM review at the orchestration/interface layer, where it can observe runtime context plus specialist proposals without belonging to any topology-owning twin.

Existing specialist agents and their twin slots are unchanged.

## Future API Hook

When an LLM API is added, it should replace only the response provider behind `LLMAdvisorAgent.propose()` or a subclass with the same `AgentProposal` output. The T-layer manifest and the twin/agent separation do not need to change.
