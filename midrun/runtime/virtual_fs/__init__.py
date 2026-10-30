"""Transport-agnostic content for Midrun's read-only virtual filesystem.

Owns the actual "what does this virtual path contain" policy -- both
`midrun.fuse.filesystem.MidrunFS` (a thin mfusepy adapter) and
`midrun.rest.app` (a thin `GET /files/{path}` adapter) read through the plain
functions here (`virtual_file_content`, `is_virtual_dir`, `list_virtual_dir`)
rather than each re-deriving the same `MidrunRuntime` state independently.
"""
