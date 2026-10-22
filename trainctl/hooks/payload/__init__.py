"""Bounded, non-evaluating description of Lightning callback arguments.

Builds the light hook's `params.json` payload (schema/identity fields plus a bounded
description of the callback's own arguments). A tensor argument is described --
shape, dtype, device -- never evaluated: no `.item()`, `.tolist()`, or `.cpu()` on
this path. Increment I3 reuses `describe_arguments` for the heavy manifest's
non-tensor structure description, alongside the separate export of the tensors
themselves.
"""
