from graph import KnowledgeGraph
from relationships import RelationshipType
from typing import Optional, List

from world import Location, Item


class Entity:
    def __init__(self) -> None:
        pass


class PhysicalAttributes:
    name: str
    sex: str
    birth: int  # Represent birth time as ticks since epoch
    death: Optional[int] = None
    race: str
    location: "Location"

    inventory: List["Item"]
    owned_items: List["Item"]
    owned_properties: List["Location"]

    # physical_condition: List[
    #     "HealthCondition"
    # ]  # list of various conditions like injury/health

    def __init__(self) -> None:
        pass


class Personality:
    def __init__(self) -> None:
        pass


class Character(Entity):
    def __init__(self) -> None:
        self.family = KnowledgeGraph()
        self.personality = Personality()
        self.physical_attributes = PhysicalAttributes()

    def add_familial_edge(self, subject: str, predicate: str, object: str) -> None:
        if predicate in [relation.value for relation in list(RelationshipType)]:
            self.family.add_edge(subject, predicate, object)
        else:
            raise ValueError(f"Invalid predicate: {predicate}")
