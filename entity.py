import uuid
from uuid import UUID
from typing import TYPE_CHECKING, List, Tuple

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

    def to_subject_predicate_object(self) -> List[Tuple[str, str, str]]:
        return []

    def render(self) -> str:
        """Default render method for entities. Returns an empty cell."""
        return "."
