"""The type of a command argument that is a zero-based index."""

from typing import Annotated

from pydantic import Field

NonNegativeIndex = Annotated[int, Field(ge=0)]
