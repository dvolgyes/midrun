"""Bootstraps a run-local, editable `hooks/` tree, optionally seeded from a baseline directory.

Owns exactly the filesystem lifecycle described in `LIFECYCLE_HOOKS_PLAN.md` sections 3 and 9:
staged construction, ownership/execution-bit preservation, disabled-example seeding, atomic
publication, and resume/conflict detection through a small external marker plus an internal
provenance record. It does not scan for enabled scripts or dispatch anything -- see
`trainctl/hooks/catalog.py` for the callback catalogue this module seeds slots for, and later
increments for discovery and execution.
"""
