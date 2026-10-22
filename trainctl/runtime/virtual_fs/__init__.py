"""Transport-agnostic content for Trainctl's read-only virtual filesystem.

Owns the actual "what does this virtual path contain" policy -- both
`trainctl.fuse.filesystem.TrainctlFS` (a thin mfusepy adapter) and
`trainctl.rest.app` (a thin `GET /files/{path}` adapter) read through the plain
functions here (`virtual_file_content`, `is_virtual_dir`, `list_virtual_dir`)
rather than each re-deriving the same `TrainctlRuntime` state independently.
"""
