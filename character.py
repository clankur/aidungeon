import random
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
        self.world = world
        location.add_occupant(self)

    def add_familial_edge(
        self,
        subject: "Character",
        predicate: RelationshipType,
        object: "Character",
    ) -> None:
        if predicate in [relation for relation in list(RelationshipType)]:
            self.world.add_edge(subject, predicate.value, object)
        else:
            raise ValueError(f"Invalid predicate: {predicate}")

    def move(self, coords: Optional[Tuple[int, int]] = None) -> None:
        curr_location = self.world.get_edges(
            predicate="is_occupied_by", object_node=self
        )
        if len(curr_location) == 0:
            raise ValueError(f"Character {self.name} has no location! {curr_location}")
        if len(curr_location) > 1:
            raise ValueError(
                f"Character {self.name} in multiple tiles! {curr_location}"
            )
        curr_tile: Tile = curr_location[0][0]
        curr_building = curr_tile.building
        valid_positions = [
            (curr_tile.local_coords[0] - 1, curr_tile.local_coords[1]),
            (curr_tile.local_coords[0] + 1, curr_tile.local_coords[1]),
            (curr_tile.local_coords[0], curr_tile.local_coords[1] - 1),
            (curr_tile.local_coords[0], curr_tile.local_coords[1] + 1),
        ]
        valid_positions = [
            pos
            for pos in valid_positions
            if pos[0] >= 0
            and pos[1] >= 0
            and pos[0] < curr_building.internal_dims[0]
            and pos[1] < curr_building.internal_dims[1]
        ]

        # Filter out positions that are already occupied
        unoccupied_positions = []
        for pos in valid_positions:
            tile = curr_building.get_tile(pos)
            if tile and not tile.get_occupants():
                unoccupied_positions.append(pos)

        # Check if there are any valid unoccupied positions
        if not unoccupied_positions:
            return  # No valid moves available

        if coords:
            # Verify the specified coordinates are valid and unoccupied
            if coords not in unoccupied_positions:
                return
        else:
            # Choose a random unoccupied position
            coords = random.choice(unoccupied_positions)

        # Find the new tile in the building
        new_tile = curr_building.get_tile(coords)
        curr_tile.remove_occupant(self)
        new_tile.add_occupant(self)

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
