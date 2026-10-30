"""Checks that MidrunMixin's keyword defaults equal MidrunConfig's field defaults."""

import dataclasses
import inspect

from midrun import MidrunMixin
from midrun.config import MidrunConfig

# The mixin prefixes the master switch, so it cannot clash with a user's own kwarg.
_MIXIN_TO_CONFIG_NAMES = {"midrun_enabled": "enabled"}


def test_mixin_keyword_defaults_equal_the_config_defaults() -> None:
    parameters = inspect.signature(MidrunMixin.__init__).parameters
    mixin_defaults = {
        _MIXIN_TO_CONFIG_NAMES.get(name, name): parameter.default
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    config_defaults = {
        field.name: field.default for field in dataclasses.fields(MidrunConfig)
    }
    assert mixin_defaults == config_defaults
