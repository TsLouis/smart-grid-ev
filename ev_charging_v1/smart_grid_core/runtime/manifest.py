from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class AgentManifest:
    """Declared contract for a subagent.

    `consumes` and `produces` are capability strings. RootAgent uses them
    to topologically order subagents — producer runs before consumer.
    Capability names are opaque to the runtime; semantics live in the
    agent and twin that produce/consume them.
    """

    agent_id: str
    agent_type: str
    consumes: Sequence[str] = field(default_factory=tuple)
    produces: Sequence[str] = field(default_factory=tuple)
    optional_inputs: Sequence[str] = field(default_factory=tuple)
    level: int = 0
    metadata: Mapping = field(default_factory=dict)
