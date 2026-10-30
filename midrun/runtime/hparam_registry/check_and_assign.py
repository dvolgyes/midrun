"""Checks a new tunable-hparam value and assigns it."""

from dataclasses import replace
from typing import Any

import torch

from midrun.runtime.hparam_registry.describe_hparam import describe_hparam
from midrun.runtime.hparam_registry.get_field import get_field
from midrun.runtime.hparam_registry.hparam_value import HparamValue
from midrun.runtime.hparam_registry.set_field import set_field
from midrun.runtime.hparam_registry.tunable_hparam_spec import TunableHparamSpec
from midrun.runtime.hparam_registry.typed_hparam_value import typed_hparam_value
from midrun.runtime.hparam_registry.validate_bounds import validate_bounds


def check_and_assign(
    spec: TunableHparamSpec, parent: Any, leaf: str, new_value: object
) -> HparamValue:
    """Checks `new_value` against `parent.<leaf>`'s current type and bounds, then assigns it.

    The value must already have the knob's type: no value is converted to another
    type, so `"false"` is not a bool and `7.9` is not an int. An int is a valid value
    for a float knob. A float value must be finite. A tensor knob takes a nested list
    of its shape, matched to the current tensor's dtype and device, and copied in
    place with `copy_` so that optimizer and autograd references to it stay valid.

    Returns:
        The new value, with `spec`'s declared bounds and label.

    Raises:
        ValueError: the value has the wrong type or shape, is not finite, violates
            `spec`'s bounds, or the current value's type is unsupported.
    """
    current = get_field(parent, leaf)
    value = typed_hparam_value(current, leaf, new_value)
    validate_bounds(spec, value)
    if isinstance(current, torch.Tensor):
        # Without no_grad, an in-place copy into an `nn.Parameter` raises.
        with torch.no_grad():
            current.copy_(value)
    else:
        set_field(parent, leaf, value)
    return replace(
        describe_hparam(get_field(parent, leaf)),
        min=spec.min,
        max=spec.max,
        label=spec.label,
    )
