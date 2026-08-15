from pydantic import BaseModel
from typing import Any


class Document(BaseModel):
    text: str
    metadata: dict[str, Any]
