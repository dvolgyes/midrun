"""Decides from an `Accept` header whether a client prefers HTML to JSON."""

import re

# The qvalue grammar of RFC 9110, section 12.4.2.
_QVALUE = re.compile(r"0(\.\d{0,3})?|1(\.0{0,3})?")


def prefers_html(accept: str) -> bool:
    """Returns whether `accept` ranks `text/html` above `application/json`.

    Only an explicit media type counts, so `*/*`, as curl and HTTP libraries send
    it, keeps JSON; a browser lists `text/html` first. A malformed q-value counts
    as 0.
    """
    weights = {"text/html": 0.0, "application/json": 0.0}
    for media_range in accept.split(","):
        media_type, *params = (part.strip() for part in media_range.split(";"))
        if media_type.lower() not in weights:
            continue
        weight = 1.0
        for param in params:
            name, _, value = param.partition("=")
            if name.strip().lower() == "q":
                weight = float(value) if _QVALUE.fullmatch(value.strip()) else 0.0
        weights[media_type.lower()] = weight
    return weights["text/html"] > weights["application/json"]
