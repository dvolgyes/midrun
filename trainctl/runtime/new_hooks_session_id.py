"""Makes a unique id for a hooks session."""

import os
import secrets
import time


def new_hooks_session_id() -> str:
    """A collision-resistant session id: `commands.next_id` is process-local only."""
    return f"{time.time_ns()}-{os.getpid()}-{secrets.token_hex(4)}"
