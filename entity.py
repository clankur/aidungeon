import uuid
from uuid import UUID
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from world import World


class Entity:
    name: str
    uuid: UUID

    def __init__(self, name: str, world: "World") -> None:
        self.name = name
        self.uuid = uuid.uuid4()
        world.add_edge(self.uuid, "uuid", self)
        world.add_edge(self.name, "name", self)
