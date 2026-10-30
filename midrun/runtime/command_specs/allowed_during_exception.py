"""The command kinds that the exception hold runs."""

from midrun.runtime.command_specs.command_specs import COMMAND_SPECS

ALLOWED_DURING_EXCEPTION = frozenset(
    kind for kind, spec in COMMAND_SPECS.items() if spec.during_exception
)
