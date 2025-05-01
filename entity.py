import uuid
from uuid import UUID


class Entity:
    name: str
    uuid: UUID

    def __init__(self, name: str) -> None:
        self.name = name
        self.uuid = uuid.uuid4()
