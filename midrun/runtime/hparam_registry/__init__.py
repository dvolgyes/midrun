"""Generic named tunable-hyperparameter registry: resolve, describe, and type-checked assign.

A LightningModule opts knobs in via a class attribute `midrun_tunable_hparams`,
in either form:

  * `list[str]` -- bare dotted paths, no bounds/label; type is auto-detected from
    the current value at read time.
  * `dict[str, dict]` -- path -> metadata dict with optional `min`/`max`/`label`.
    The dict key IS the resolution path (same role the bare string plays in the
    list form); a display name is `label`, not a second key scheme.

A path segment may be an attribute or a dict key (`get_field`/`set_field`
resolve either), so a path can walk into a nested dict-of-dicts, not just plain
attribute chains. Declarations are merged across `type(pl_module).__mro__` via
`collect_tunable_hparams`, so a mixin and a subclass can each contribute their
own fields without one silently shadowing the other's whole declaration; a
more-derived class re-declaring the same path replaces that path's spec
wholesale. Everything not listed stays exactly as read-only as
`MidrunRuntime.model_info.hparams` already makes it.
"""
