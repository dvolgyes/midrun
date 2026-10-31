"""The command kinds that run only while a batch is in flight."""

from midrun.runtime.command_specs.command_specs import COMMAND_SPECS

NEEDS_BATCH = frozenset(
    kind for kind, spec in COMMAND_SPECS.items() if spec.needs_batch
)
