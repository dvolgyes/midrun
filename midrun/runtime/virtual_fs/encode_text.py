"""Encodes the text content of a virtual file."""


def encode_text(value: str) -> bytes:
    """Returns `value` as UTF-8 bytes that end with one newline."""
    return (value if value.endswith("\n") else value + "\n").encode()
