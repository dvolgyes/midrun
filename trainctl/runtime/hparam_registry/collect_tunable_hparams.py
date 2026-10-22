"""Collects the tunable-hparam declarations of a model class."""

from trainctl.runtime.hparam_registry.tunable_hparam_spec import TunableHparamSpec


def collect_tunable_hparams(model_cls: type) -> dict[str, TunableHparamSpec]:
    """Merges `trainctl_tunable_hparams` across `model_cls.__mro__` into one spec dict.

    Walks the MRO ancestors-first (`reversed(model_cls.__mro__)`), reading only
    each class's own directly-declared attribute (`base.__dict__`, never plain
    `getattr`, so a value already visible through inheritance isn't reprocessed
    at every MRO level it's visible from). A more-derived class's entry for a
    given path fully replaces a less-derived one's -- whole-spec replacement,
    matching ordinary attribute-shadowing semantics, not a per-key metadata
    merge. Each class may use either accepted form, independently of the others.

    Raises:
        TypeError: a class's `trainctl_tunable_hparams` is a bare `str` (missing
            the enclosing list brackets -- iterating it would otherwise silently
            register one bogus single-character path per letter), or a dict
            form's per-path metadata value is not itself a `dict`.
    """
    result: dict[str, TunableHparamSpec] = {}
    for base in reversed(model_cls.__mro__):
        declared = base.__dict__.get("trainctl_tunable_hparams")
        if declared is None:
            continue
        if isinstance(declared, str):
            raise TypeError(
                f"{base.__name__}.trainctl_tunable_hparams must be a list[str] or "
                f"dict[str, dict], got a bare str {declared!r} "
                "(missing the enclosing list brackets?)"
            )
        if isinstance(declared, dict):
            for path, meta in declared.items():
                if not isinstance(meta, dict):
                    raise TypeError(
                        f"{base.__name__}.trainctl_tunable_hparams[{path!r}] must "
                        f"be a dict, got {type(meta).__name__}"
                    )
                result[path] = TunableHparamSpec(
                    path=path,
                    min=meta.get("min"),
                    max=meta.get("max"),
                    label=meta.get("label"),
                )
        else:
            for path in declared:
                result[path] = TunableHparamSpec(path=path)
    return result
