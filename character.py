import random
from graph import KnowledgeGraph
from typing import Optional, List, Tuple
from item import Entity, Item
from world import Building, World, Tile
from entity import Entity
import json
from enums import get_character_emoji, RelationshipType, Direction


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
        if not hasattr(location, "add_occupant"):
            raise TypeError(
                f"Location for character {name} must be a Tile object, got {type(location)}"
            )
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

    def move(self, direction: Direction) -> bool:
        """Moves the character one tile in the specified direction if valid."""
        curr_location_edges = self.world.get_edges(
            predicate="is_occupied_by", object_node=self
        )
        if not curr_location_edges:
            return False

        curr_tile_entity = curr_location_edges[0][0]

        if not (
            hasattr(curr_tile_entity, "local_coords")
            and hasattr(curr_tile_entity, "building")
        ):
            return False

        curr_tile: "Tile" = curr_tile_entity
        curr_building: "Building" = curr_tile.building

        dx, dy = 0, 0
        if direction == Direction.NORTH:
            dy = -1
        elif direction == Direction.SOUTH:
            dy = 1
        elif direction == Direction.WEST:
            dx = -1
        elif direction == Direction.EAST:
            dx = 1
        else:
            return False

        target_coords = (curr_tile.local_coords[0] + dx, curr_tile.local_coords[1] + dy)

        if not (
            0 <= target_coords[0] < curr_building.internal_dims[0]
            and 0 <= target_coords[1] < curr_building.internal_dims[1]
        ):
            return False

        target_tile = curr_building.get_tile(target_coords)
        if not target_tile:
            return False

        for occupant in target_tile.get_occupants():
            if isinstance(occupant, Character):
                return False

        try:
            curr_tile.remove_occupant(self)
            target_tile.add_occupant(self)
            return True
        except ValueError as e:
            return False

    def chat(self, target_character: "Character", message: str) -> bool:
        """Allows the character to chat with an adjacent character."""
        if not isinstance(target_character, Character):
            return False

        if self == target_character:
            return False

        self_location_edges = self.world.get_edges(
            predicate="is_occupied_by", object_node=self
        )
        if not self_location_edges:
            return False
        self_tile_entity = self_location_edges[0][0]
        if not (
            hasattr(self_tile_entity, "local_coords")
            and hasattr(self_tile_entity, "building")
        ):
            return False
        self_tile: "Tile" = self_tile_entity

        target_location_edges = self.world.get_edges(
            predicate="is_occupied_by", object_node=target_character
        )
        if not target_location_edges:
            return False
        target_tile_entity = target_location_edges[0][0]
        if not (
            hasattr(target_tile_entity, "local_coords")
            and hasattr(target_tile_entity, "building")
        ):
            return False
        target_tile: "Tile" = target_tile_entity

        if self_tile.building.uuid != target_tile.building.uuid:
            return False

        dx = abs(self_tile.local_coords[0] - target_tile.local_coords[0])
        dy = abs(self_tile.local_coords[1] - target_tile.local_coords[1])

        if (dx == 1 and dy == 0) or (dx == 0 and dy == 1):
            timestamp = self.world.get_current_world_time()
            # TODO: implement chat functionality as an event
            return True
        else:
            return False

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
