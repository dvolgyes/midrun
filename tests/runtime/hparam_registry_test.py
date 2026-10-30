"""Behavioral tests for the generic tunable-hyperparameter registry.

`resolve_hparam_path`/`describe_hparam`/`check_and_assign` are plain functions over arbitrary
parent/leaf attribute pairs, independent of `MidrunRuntime` --
`tests/runtime/set_hparam_command_test.py` covers the end-to-end command path.
"""

import pytest
import torch

from midrun.runtime.hparam_registry.check_and_assign import check_and_assign
from midrun.runtime.hparam_registry.collect_tunable_hparams import (
    collect_tunable_hparams,
)
from midrun.runtime.hparam_registry.describe_hparam import (
    MAX_INLINE_TENSOR_ELEMENTS,
    describe_hparam,
)
from midrun.runtime.hparam_registry.get_field import get_field
from midrun.runtime.hparam_registry.hparam_value import HparamValue
from midrun.runtime.hparam_registry.resolve_hparam_path import resolve_hparam_path
from midrun.runtime.hparam_registry.set_field import set_field
from midrun.runtime.hparam_registry.tunable_hparam_spec import TunableHparamSpec
from midrun.runtime.hparam_registry.validate_bounds import validate_bounds


class _Hparams:
    def __init__(self) -> None:
        self.augmentation_strength = 0.5


class _Model:
    def __init__(self) -> None:
        self.hparams = _Hparams()
        self.flag = True
        self.label = "baseline"
        self.weight = torch.tensor([1.0, 2.0, 3.0])


class _Trainer:
    def __init__(self, datamodule: object) -> None:
        self.datamodule = datamodule


def test_resolve_plain_attribute_path() -> None:
    model = _Model()

    parent, leaf = resolve_hparam_path(model, trainer=None, path="flag")

    assert parent is model
    assert leaf == "flag"


def test_resolve_nested_pl_module_path() -> None:
    model = _Model()

    parent, leaf = resolve_hparam_path(
        model, trainer=None, path="hparams.augmentation_strength"
    )

    assert parent is model.hparams
    assert leaf == "augmentation_strength"


def test_resolve_trainer_rooted_path() -> None:
    model = _Model()
    trainer = _Trainer(datamodule=model)

    parent, leaf = resolve_hparam_path(model, trainer, path="trainer.datamodule.flag")

    assert parent is model
    assert leaf == "flag"


def test_describe_bool_before_int_ordering() -> None:
    result = describe_hparam(True)

    assert result.type == "bool"
    assert result.value is True


def test_describe_int() -> None:
    assert describe_hparam(3).type == "int"


def test_describe_float() -> None:
    assert describe_hparam(0.5).type == "float"


def test_describe_str() -> None:
    assert describe_hparam("baseline").type == "str"


def test_describe_tensor_reports_dtype_device_and_shape() -> None:
    tensor = torch.tensor([1.0, 2.0, 3.0])

    result = describe_hparam(tensor)

    assert result.type == "tensor"
    assert result.dtype == "torch.float32"
    assert result.device == "cpu"
    assert result.shape == [3]
    assert result.value == [1.0, 2.0, 3.0]


def test_describe_oversized_tensor_omits_value_but_keeps_metadata() -> None:
    size = MAX_INLINE_TENSOR_ELEMENTS + 1
    tensor = torch.zeros(size)

    result = describe_hparam(tensor)

    assert result.value is None
    assert result.shape == [size]
    assert result.dtype == "torch.float32"


def test_describe_unsupported_type_raises() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        describe_hparam([1, 2, 3])


def _assign(parent: object, leaf: str, value: object) -> HparamValue:
    spec = TunableHparamSpec(path=leaf)
    return check_and_assign(spec, parent, leaf, value)


def _current(model: _Model, leaf: str) -> object:
    value = getattr(model, leaf)
    return value.clone() if isinstance(value, torch.Tensor) else value


@pytest.mark.parametrize(
    ("leaf", "value"),
    [
        ("flag", 0),
        ("flag", "false"),
        ("count", "5"),
        ("count", 7.9),
        ("count", True),
        ("label", 123),
        ("label", {"a": 1}),
        ("weight", "abc"),
        ("weight", True),
    ],
)
def test_check_and_assign_rejects_a_value_of_another_type(
    leaf: str, value: object
) -> None:
    model = _Model()
    model.count = 1
    before = _current(model, leaf)

    with pytest.raises(ValueError, match="hparam; got"):
        _assign(model, leaf, value)

    after = getattr(model, leaf)
    if isinstance(before, torch.Tensor):
        assert torch.equal(after, before)
    else:
        assert after == before


def test_check_and_assign_bool() -> None:
    model = _Model()

    result = _assign(model, "flag", False)

    assert result.type == "bool"
    assert result.value is False
    assert model.flag is False


def test_check_and_assign_int() -> None:
    model = _Model()
    model.count = 1

    result = _assign(model, "count", 5)

    assert result.type == "int"
    assert result.value == 5


def test_check_and_assign_float_accepts_an_int() -> None:
    model = _Model()

    result = _assign(model.hparams, "augmentation_strength", 1)

    assert result.type == "float"
    assert model.hparams.augmentation_strength == 1.0
    assert isinstance(model.hparams.augmentation_strength, float)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_check_and_assign_float_rejects_a_non_finite_value(value: float) -> None:
    model = _Model()

    with pytest.raises(ValueError, match="finite"):
        _assign(model.hparams, "augmentation_strength", value)

    assert model.hparams.augmentation_strength == 0.5


def test_check_and_assign_str() -> None:
    model = _Model()

    result = _assign(model, "label", "candidate")

    assert result.type == "str"
    assert result.value == "candidate"


def test_check_and_assign_tensor_matches_dtype_moves_device_and_preserves_identity() -> (
    None
):
    model = _Model()
    original = model.weight

    result = _assign(model, "weight", [4.0, 5.0, 6.0])

    assert model.weight is original
    assert torch.equal(model.weight, torch.tensor([4.0, 5.0, 6.0]))
    assert result.dtype == "torch.float32"
    assert result.device == "cpu"


def test_check_and_assign_tensor_shape_mismatch_raises() -> None:
    model = _Model()

    with pytest.raises(ValueError, match="shape mismatch"):
        _assign(model, "weight", [1.0, 2.0])


def test_check_and_assign_tensor_rejects_nan_elements() -> None:
    model = _Model()

    with pytest.raises(ValueError, match="finite"):
        _assign(model, "weight", [1.0, float("nan"), 3.0])

    assert torch.equal(model.weight, torch.tensor([1.0, 2.0, 3.0]))


def test_check_and_assign_checks_the_type_before_the_bounds() -> None:
    model = _Model()
    spec = TunableHparamSpec(path="augmentation_strength", min=0, max=1)

    with pytest.raises(ValueError, match="is a float hparam; got str"):
        check_and_assign(spec, model.hparams, "augmentation_strength", "0.7")


def test_check_and_assign_unsupported_current_type_raises() -> None:
    model = _Model()
    model.unsupported = object()

    with pytest.raises(ValueError, match="unsupported"):
        _assign(model, "unsupported", 1)


# ---- collect_tunable_hparams: MRO merge ------------------------------------


class _MixinA:
    midrun_tunable_hparams = ["a_field"]


class _MixinB:
    midrun_tunable_hparams = ["b_field"]


class _CombinedListForm(_MixinA, _MixinB):
    pass


def test_collect_tunable_hparams_merges_bare_list_across_mro() -> None:
    specs = collect_tunable_hparams(_CombinedListForm)

    assert set(specs) == {"a_field", "b_field"}
    assert specs["a_field"] == TunableHparamSpec(path="a_field")
    assert specs["b_field"] == TunableHparamSpec(path="b_field")


class _DictForm:
    midrun_tunable_hparams = {
        "hparams.lr": {"min": 0.0, "max": 1.0, "label": "Learning rate"},
    }


def test_collect_tunable_hparams_dict_form_carries_min_max_label() -> None:
    specs = collect_tunable_hparams(_DictForm)

    assert specs["hparams.lr"] == TunableHparamSpec(
        path="hparams.lr", min=0.0, max=1.0, label="Learning rate"
    )


class _AncestorBounded:
    midrun_tunable_hparams = {"shared": {"min": 0, "max": 10, "label": "ancestor"}}


class _LeafReplaces(_AncestorBounded):
    midrun_tunable_hparams = {"shared": {"label": "leaf"}}


def test_collect_tunable_hparams_leaf_replaces_ancestor_spec_wholesale() -> None:
    specs = collect_tunable_hparams(_LeafReplaces)

    assert specs["shared"] == TunableHparamSpec(path="shared", label="leaf")


class _MixinListForm:
    midrun_tunable_hparams = ["mixin_field"]


class _SubclassDictForm(_MixinListForm):
    midrun_tunable_hparams = {"dict_field": {"label": "Dict field"}}


def test_collect_tunable_hparams_mixes_list_and_dict_forms_across_mro() -> None:
    specs = collect_tunable_hparams(_SubclassDictForm)

    assert specs["mixin_field"] == TunableHparamSpec(path="mixin_field")
    assert specs["dict_field"] == TunableHparamSpec(
        path="dict_field", label="Dict field"
    )


class _BareStrForm:
    midrun_tunable_hparams = "oops"


def test_collect_tunable_hparams_bare_str_raises_type_error() -> None:
    with pytest.raises(TypeError, match="brackets"):
        collect_tunable_hparams(_BareStrForm)


class _NonDictMetadataForm:
    midrun_tunable_hparams = {"field": "not-a-dict"}


def test_collect_tunable_hparams_non_dict_metadata_raises_type_error() -> None:
    with pytest.raises(TypeError, match="field"):
        collect_tunable_hparams(_NonDictMetadataForm)


# ---- dict-nested paths (get_field/set_field/resolve/check_and_assign) -----


class _NestedModel:
    def __init__(self) -> None:
        self.config = {"optimizer": {"lr_schedule": {"warmup_steps": 100}}}
        self.state = {"weight": torch.tensor([1.0, 2.0])}


def test_resolve_walks_into_nested_dict() -> None:
    model = _NestedModel()

    parent, leaf = resolve_hparam_path(
        model, trainer=None, path="config.optimizer.lr_schedule.warmup_steps"
    )

    assert parent is model.config["optimizer"]["lr_schedule"]
    assert leaf == "warmup_steps"


def test_check_and_assign_reads_and_writes_dict_nested_leaf() -> None:
    model = _NestedModel()
    inner = model.config["optimizer"]["lr_schedule"]

    result = _assign(inner, "warmup_steps", 200)

    assert result.value == 200
    assert inner["warmup_steps"] == 200


def test_check_and_assign_preserves_dict_nested_tensor_identity_dtype_device() -> None:
    model = _NestedModel()
    original = model.state["weight"]

    result = _assign(model.state, "weight", [3.0, 4.0])

    assert model.state["weight"] is original
    assert torch.equal(model.state["weight"], torch.tensor([3.0, 4.0]))
    assert result.dtype == "torch.float32"
    assert result.device == "cpu"


def test_get_field_raises_when_key_is_neither_attribute_nor_mapping_key() -> None:
    with pytest.raises(AttributeError):
        get_field({"a": 1}, "b")


def test_set_field_raises_when_key_is_neither_attribute_nor_mapping_key() -> None:
    with pytest.raises(AttributeError):
        set_field(object(), "missing", 1)


# ---- validate_bounds --------------------------------------------------------


def test_validate_bounds_noop_when_unbounded() -> None:
    spec = TunableHparamSpec(path="x")

    validate_bounds(spec, 999)  # must not raise


def test_validate_bounds_scalar_within_range_passes() -> None:
    spec = TunableHparamSpec(path="x", min=0, max=10)

    validate_bounds(spec, 5)  # must not raise


def test_validate_bounds_scalar_below_min_raises() -> None:
    spec = TunableHparamSpec(path="x", min=0, max=10)

    with pytest.raises(ValueError, match=">= 0"):
        validate_bounds(spec, -1)


def test_validate_bounds_scalar_above_max_raises() -> None:
    spec = TunableHparamSpec(path="x", min=0, max=10)

    with pytest.raises(ValueError, match="<= 10"):
        validate_bounds(spec, 11)


def test_validate_bounds_rank0_tensor_behaves_like_scalar() -> None:
    spec = TunableHparamSpec(path="x", min=0, max=10)

    validate_bounds(spec, torch.tensor(5.0))  # must not raise
    with pytest.raises(ValueError, match=">= 0"):
        validate_bounds(spec, torch.tensor(-1.0))


def test_validate_bounds_multi_element_tensor_rejects_any_out_of_range_element() -> (
    None
):
    spec = TunableHparamSpec(path="x", min=0, max=10)

    with pytest.raises(ValueError, match=">= 0"):
        validate_bounds(spec, [5.0, -1.0])


def test_validate_bounds_multi_element_tensor_all_in_range_passes() -> None:
    spec = TunableHparamSpec(path="x", min=0, max=10)

    validate_bounds(spec, torch.tensor([1.0, 5.0, 9.0]))  # must not raise


@pytest.mark.parametrize(
    ("current", "value"),
    [
        (torch.tensor([1, 2]), [7.9, 2.5]),
        (torch.tensor([1.0, 2.0]), [True, False]),
        (torch.tensor([1, 2]), [True, False]),
        (torch.tensor([True, False]), [1, 0]),
        (torch.tensor([True, False]), [1.0, 0.0]),
    ],
)
def test_check_and_assign_tensor_rejects_elements_of_another_type(
    current: torch.Tensor, value: list
) -> None:
    model = _Model()
    model.weight = current
    before = current.clone()

    with pytest.raises(ValueError, match="hparam; got"):
        _assign(model, "weight", value)

    assert torch.equal(model.weight, before)


def test_check_and_assign_float_tensor_accepts_int_elements() -> None:
    model = _Model()

    _assign(model, "weight", [4, 5, 6])

    assert torch.equal(model.weight, torch.tensor([4.0, 5.0, 6.0]))
    assert model.weight.dtype == torch.float32


def test_check_and_assign_updates_a_parameter_in_place() -> None:
    model = _Model()
    model.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    original = model.weight

    result = _assign(model, "weight", [4.0, 5.0, 6.0])

    assert model.weight is original
    assert torch.equal(model.weight.detach(), torch.tensor([4.0, 5.0, 6.0]))
    assert model.weight.requires_grad
    assert result.value == [4.0, 5.0, 6.0]


@pytest.mark.parametrize("key", ["values", "items", "update"])
def test_a_mapping_key_named_like_a_dict_method_is_read_and_written(key: str) -> None:
    model = _NestedModel()
    model.config = {key: {"warmup_steps": 100}, "schedule": {key: 1}}

    parent, leaf = resolve_hparam_path(
        model, trainer=None, path=f"config.{key}.warmup_steps"
    )
    result = _assign(model.config["schedule"], key, 2)

    assert parent is model.config[key]
    assert leaf == "warmup_steps"
    assert result.value == 2
    assert model.config["schedule"][key] == 2
