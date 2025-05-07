from graph import KnowledgeGraph
from typing import Optional, List, Tuple
from item import Entity, Item
from world import Building, World, Tile
from entity import Entity
import json
from enums import get_character_emoji, RelationshipType


class PhysicalAttributes:
    sex: str
    birth: int  # Represent birth time as ticks since epoch
    death: Optional[int] = None
    race: str  # TODO: consider Enum

    inventory: List["Item"]
    owned_items: List["Item"]
    owned_properties: List["Building"]

    # physical_condition: List[
    #     "HealthCondition"
    # ]  # TODO: list of various conditions like injury/health

    def __init__(self, sex: str, birth: int, race: str) -> None:
        self.sex = sex
        self.birth = birth
        self.race = race
        self.inventory = []
        self.owned_items = []
        self.owned_properties = []

    def __repr__(self) -> str:
        data = {
            "sex": self.sex,
            "birth": self.birth,
            "death": self.death,
            "race": self.race,
            "inventory": [str(item) for item in self.inventory],
            "owned_items": [str(item) for item in self.owned_items],
            "owned_properties": [str(loc) for loc in self.owned_properties],
        }
        return json.dumps(data)

    def to_predicate_object(self) -> List[Tuple[str, str, str]]:
        return [
            ("sex", self.sex),
            ("birth", self.birth),
            ("race", self.race),
        ]


class Personality:
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return json.dumps({})

    def to_predicate_object(self) -> List[Tuple[str, str, str]]:
        return []


class Character(Entity):
    def __init__(
        self,
        name: str,
        sex: str,
        race: str,
        location: "Tile",
        world: "World",
        birth_time: Optional[int] = None,
    ) -> None:
        super().__init__(name, world)
        actual_birth_time = (
            birth_time if birth_time is not None else world.get_current_world_time()
        )
        self.personality = Personality()
        self.physical_attributes = PhysicalAttributes(
            sex=sex, birth=actual_birth_time, race=race
        )
        location.add_occupant(self)

    def add_familial_edge(
        self,
        world_graph: "KnowledgeGraph",
        subject: "Character",
        predicate: RelationshipType,
        object: "Character",
    ) -> None:
        if predicate in [relation for relation in list(RelationshipType)]:
            world_graph.add_edge(subject, predicate.value, object)
        else:
            raise ValueError(f"Invalid predicate: {predicate}")

    def __repr__(self) -> str:
        data = {
            "name": self.name,
            "personality": json.loads(repr(self.personality)),
            "physical_attributes": json.loads(repr(self.physical_attributes)),
        }
        return json.dumps(data)

    def to_subject_predicate_object(self) -> List[Tuple[str, str, str]]:
        personality = [
            (self.name, trait, personality)
            for trait, personality in self.personality.to_predicate_object()
        ]
        physical_attributes = [
            (self.name, trait, attribute)
            for trait, attribute in self.physical_attributes.to_predicate_object()
        ]
        # TODO: for disambiguation consider bundling UUID with name
        return personality + physical_attributes

    def render(self) -> str:
        """Returns the character's emoji based on their race and sex."""
        return get_character_emoji(
            self.physical_attributes.race, self.physical_attributes.sex
        )
