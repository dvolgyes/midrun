"""Behavioral tests for TrainctlOptions, the typed TrainctlMixin keyword arguments."""

import inspect
from typing import get_type_hints

from trainctl import TrainctlMixin, TrainctlOptions


def test_options_match_the_mixin_keyword_arguments_and_types() -> None:
    parameters = inspect.signature(TrainctlMixin.__init__).parameters
    keyword_only = [
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    mixin_hints = get_type_hints(TrainctlMixin.__init__)
    assert get_type_hints(TrainctlOptions) == {
        name: mixin_hints[name] for name in keyword_only
    }
