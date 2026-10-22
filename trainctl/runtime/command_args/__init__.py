"""Per-kind argument models for queued commands, checked when a command is submitted.

The training thread runs a command long after REST accepted it, so every argument
check that does not need the live model happens here. A handler then reads its
arguments without casts or defaults of its own.
"""
