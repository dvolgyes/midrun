"""On-demand CUDA memory snapshots and bounded allocator-history recording.

Orchestrates `torch.cuda.memory`'s own snapshot/history APIs -- no allocator
reimplementation. `_record_memory_history`'s enabled/disabled flag is process-global
torch state, not per-runtime, so callers (tests included) must restore it afterward.
"""
