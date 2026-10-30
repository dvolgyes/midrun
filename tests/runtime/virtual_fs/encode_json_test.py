"""Behavioral tests: virtual JSON files are standard JSON, as the REST API's are."""

import json
from dataclasses import dataclass

from midrun.runtime.virtual_fs.encode_json import encode_json


def _strict_loads(encoded: bytes) -> object:
    def reject(constant: str) -> None:
        raise ValueError(f"non-standard JSON constant {constant}")

    return json.loads(encoded, parse_constant=reject)


@dataclass
class _Metric:
    value: float


def test_non_finite_floats_encode_as_null() -> None:
    encoded = encode_json(
        {"nan": float("nan"), "values": [float("inf"), -float("inf"), 0.5]}
    )

    assert _strict_loads(encoded) == {"nan": None, "values": [None, None, 0.5]}


def test_a_dataclass_encodes_as_an_object_with_non_finite_floats_as_null() -> None:
    assert _strict_loads(encode_json({"loss": _Metric(float("nan"))})) == {
        "loss": {"value": None}
    }


class _Opaque:
    def __str__(self) -> str:
        return "opaque"


def test_other_values_encode_as_their_string() -> None:
    assert _strict_loads(encode_json({"value": _Opaque()})) == {"value": "opaque"}
