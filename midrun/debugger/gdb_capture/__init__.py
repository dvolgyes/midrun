"""Higher-risk, equally-supported capture mode: runs arbitrary Python inside a live target process via GDB call-injection.

Neither this project's nor a typical system CPython ships the DWARF-based
`python-gdb.py` extension needed for the standard `py-bt`/`py-list` gdb commands.
This exploits a different fact instead: a live CPython binary's dynamic symbol
table still exports its public C-API (`PyRun_SimpleString`, `Py_AddPendingCall`,
...) even fully stripped, and gdb can call these by address via
`-data-evaluate-expression`.

Safety history (do not relax without re-reading this): an earlier variant used
`PyGILState_Ensure()` + `PyRun_SimpleString(...)` directly, which permanently
deadlocked a live CUDA training process -- PyTorch's `.item()` releases the GIL
during a blocking CUDA sync, and gdb's `call` hijacking the current thread's
execution context there can deadlock; a stray `call` landing mid-`backward()`
also tripped up autograd engine locking. `Py_AddPendingCall` is markedly safer --
it schedules `func(arg)` for the interpreter's own next eval-loop checkpoint
instead of hijacking in-flight execution -- but is NOT proven fully immune: a
repeat test still deadlocked a live CUDA target once. Arming a midrun hold
first (see `midrun.debugger.hold_client`) and confirming it is active converts
this from an unrecoverable deadlock into a recoverable, retry-able reliability
issue -- not a full fix, and no help at all against an already-hung process.
Prefer `midrun.debugger.pyspy_capture` whenever a read-only stack/locals dump
is enough; reach for this module when you need to mutate live state or call an
arbitrary live function instead.
"""
