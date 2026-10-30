"""Bridges Lightning's standard-library logging into Loguru for one training session.

Bridges only the namespaces midrun itself cares about (`lightning.pytorch`,
`pytorch_lightning`, `lightning.fabric`); it never touches the root logger unless
`install_root_bridge` is called explicitly. A session's run-file sink records the
records bound to that session (`LoggingSession.bind()`) and the unbound records of
`midrun.*` modules; it rejects the records bound to a different session.
"""
