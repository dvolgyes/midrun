"""Behavioral tests for MidrunOptions, the typed MidrunMixin keyword arguments."""

import inspect
from typing import get_type_hints

from midrun import MidrunMixin, MidrunOptions


def test_options_match_the_mixin_keyword_arguments_and_types() -> None:
    parameters = inspect.signature(MidrunMixin.__init__).parameters
    keyword_only = [
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    mixin_hints = get_type_hints(MidrunMixin.__init__)
    assert get_type_hints(MidrunOptions) == {
        name: mixin_hints[name] for name in keyword_only
    }
