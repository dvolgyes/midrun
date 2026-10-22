"""Per-kind argument models for queued commands, checked when a command is submitted.

The training thread runs a command long after REST accepted it, so every argument
check that does not need the live model happens here. A handler then reads its
arguments without casts or defaults of its own.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from trainctl.runtime.artifacts import ARTIFACT_NAME_PATTERN

_Index = Annotated[int, Field(ge=0)]


class InvalidCommandError(ValueError):
    """A submitted command has an unknown kind or invalid args."""


class _Args(BaseModel):
    """Base of the argument models: no extra keys, no coercion, finite floats only."""

    # strict: a JSON "false" stays a string and 7.9 stays a float, so neither is coerced.
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False
    )


class _NoArgs(_Args):
    """Args of a command that takes none."""


class _HoldArgs(_Args):
    """Args of `hold`."""

    reason: str = ""


class _CheckpointArgs(_Args):
    """Args of `checkpoint`."""

    label: Annotated[str, Field(pattern=ARTIFACT_NAME_PATTERN)] = "manual"


class _SnapshotArgs(_Args):
    """Args of `snapshot`."""

    kind: Literal[
        "runtime", "model_summary", "optimizer_summary", "scheduler_summary"
    ] = "runtime"


class _SetLearningRateArgs(_Args):
    """Args of `set_learning_rate`."""

    value: Annotated[float, Field(ge=0)]
    optimizer: _Index = 0
    param_group: Literal["*"] | _Index = "*"


class _ParameterSelector(_Args):
    """A `reset_momentum` selector for one parameter by name."""

    parameter: str


class _ParameterPrefixSelector(_Args):
    """A `reset_momentum` selector for the parameters whose name has a prefix."""

    parameter_prefix: str


class _ParamGroupSelector(_Args):
    """A `reset_momentum` selector for one param group."""

    param_group: _Index


class _ResetMomentumArgs(_Args):
    """Args of `reset_momentum`."""

    optimizer: _Index = 0
    selector: (
        Literal["all"]
        | _ParameterSelector
        | _ParameterPrefixSelector
        | _ParamGroupSelector
    ) = "all"


class _SetDataloaderFiniteCheckArgs(_Args):
    """Args of `set_dataloader_finite_check`."""

    enabled: bool


class _SetHparamArgs(_Args):
    """Args of `set_hparam`."""

    # The knob's current value decides the valid type of `value`; see
    # `hparam_registry.check_and_assign`.
    name: str
    value: Any


class _ProfileArgs(_Args):
    """Args of `profile`."""

    warmup_steps: _Index = 0
    active_steps: Annotated[int, Field(ge=1)] = 1
    level: Literal["basic", "shapes", "memory", "full"] = "basic"
    then_hold: bool = False


class _SetLightningProfilerArgs(_Args):
    """Args of `set_lightning_profiler`."""

    level: Literal["off", "simple", "advanced", "pytorch"] = "off"


class _TorchvizCaptureArgs(_Args):
    """Args of `torchviz_capture`."""

    # None: on in a live capture, off in an exception hold.
    show_attrs: bool | None = None
    show_saved: bool | None = None


COMMAND_ARGS: Mapping[str, type[_Args]] = MappingProxyType(
    {
        "hold": _HoldArgs,
        "stop": _NoArgs,
        "checkpoint": _CheckpointArgs,
        "snapshot": _SnapshotArgs,
        "set_learning_rate": _SetLearningRateArgs,
        "reset_momentum": _ResetMomentumArgs,
        "set_dataloader_finite_check": _SetDataloaderFiniteCheckArgs,
        "set_hparam": _SetHparamArgs,
        "save_current_batch": _NoArgs,
        "save_current_loss": _NoArgs,
        "save_partial_gradients": _NoArgs,
        "save_optimizer_summary": _NoArgs,
        "release_exception": _NoArgs,
        "torchinfo_capture": _NoArgs,
        "profile": _ProfileArgs,
        "torchview_capture": _NoArgs,
        "torchviz_capture": _TorchvizCaptureArgs,
        "torchlens_capture": _NoArgs,
        "set_lightning_profiler": _SetLightningProfilerArgs,
    }
)


def validate(kind: str, args: dict[str, Any]) -> dict[str, Any]:
    """Checks `args` for a `kind` command and returns them with defaults filled in.

    Raises:
        InvalidCommandError: `kind` is unknown, or `args` has a missing, unknown, or invalid
            entry. The message names each failed argument.
    """
    model = COMMAND_ARGS.get(kind)
    if model is None:
        raise InvalidCommandError(
            f"unknown command kind {kind!r}; known kinds: {', '.join(sorted(COMMAND_ARGS))}"
        )
    try:
        checked: dict[str, Any] = model.model_validate(args).model_dump()
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'args'}: {error['msg']}"
            for error in exc.errors()
        )
        raise InvalidCommandError(f"invalid args for {kind!r}: {problems}") from None
    return checked
