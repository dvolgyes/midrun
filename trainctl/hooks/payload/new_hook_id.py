"""Makes an identifier for one hook occurrence or invocation."""

import secrets


def new_hook_id() -> str:
    """Returns a short, collision-resistant identifier for one occurrence or invocation."""
    return secrets.token_hex(8)
