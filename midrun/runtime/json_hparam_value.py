"""Normalize hyperparameter values for JSON output."""


def json_hparam_value(value: object) -> object:
    """Keep UTF-8 text readable and encode binary hyperparameter values as tagged hex."""
    if isinstance(value, bytes | bytearray):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {"$midrun_bytes_hex": value.hex()}
    if isinstance(value, dict):
        return {key: json_hparam_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_hparam_value(item) for item in value]
    return value
