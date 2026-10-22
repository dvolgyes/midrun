"""Bridges Lightning's standard-library logging into Loguru for one training session.

Bridges only the namespaces trainctl itself cares about (`lightning.pytorch`,
`pytorch_lightning`, `lightning.fabric`); it never touches the root logger unless
`install_root_bridge` is called explicitly. Only trainctl code that binds a session's
logger (`LoggingSession.bind()`) is routed into that session's run-file sink --
pre-existing, unbound `trainctl.*` diagnostics are unaffected by this module.
"""
