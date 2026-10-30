"""The command kinds that a gate hold runs."""

from midrun.runtime.command_specs.command_specs import COMMAND_SPECS

ALLOWED_DURING_HOLD = frozenset(
    kind for kind, spec in COMMAND_SPECS.items() if spec.during_hold
)
