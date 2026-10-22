"""Typed commands, their `when` scheduling spec, and the thread-safe command queue.

REST never mutates the Trainer/model directly (see the module docstring in
`trainctl.runtime.runtime`); it only enqueues a `Command` here. The training thread
evaluates which commands are due at each safe point it passes through and executes
their handlers itself.
"""
