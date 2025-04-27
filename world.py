from enum import Enum
import random


class TerrainType(Enum):
    FOREST = "forest"
    MOUNTAIN = "mountain"
    DESERT = "desert"
    CITY = "city"
    WATER = "water"
    PLAIN = "plain"
    SWAMP = "swamp"
    TUNDRA = "tundra"
    JUNGLE = "jungle"
    HILLS = "hills"


class Location:
    """Represents a location using a hierarchical grid system."""

    subregion_x: int
    subregion_y: int
    terrain: TerrainType
    # entities: list[Entity] TODO
    # buildings: list[Building] TODO

    def __init__(
        self,
        subregion_x: int,
        subregion_y: int,
        terrain: TerrainType,
    ) -> None:
        """
        Initializes a Location.

        Args:
            subregion_x: The x-coordinate within the region's subgrid.
            subregion_y: The y-coordinate within the region's subgrid.
            terrain: The type of terrain at this location.
        """
        self.subregion_x = subregion_x
        self.subregion_y = subregion_y
        self.terrain = terrain

    def __repr__(self) -> str:
        return (
            f"Location("
            f"Subregion: ({self.subregion_x}, {self.subregion_y}), "
            f"Terrain: {self.terrain.value})"
        )


class Region:
    """Represents a region of the world."""

    dims: tuple[int, int]
    locations: list[list[Location]]

    def __init__(self, dims: tuple[int, int]) -> None:
        self.dims = dims
        self.locations = [
            [Location(x, y, random.choice(list(TerrainType))) for x in range(dims[0])]
            for y in range(dims[1])
        ]

    def __repr__(self) -> str:
        return f"Region(dims={self.dims}, locations={self.locations})"


class ItemType(Enum):
    APPLE = ("apple", 5, 1)
    BREAD = ("bread", 10, 2)
    WATER = ("water", 3, 1)
    WOOD = ("wood", 2, 5)
    STONE = ("stone", 1, 5)
    IRON = ("iron", 20, 10)

    def __init__(self, item_name: str, value: int, weight: int) -> None:
        self._item_name = item_name
        self._value = value
        self._weight = weight

    def __repr__(self) -> str:
        return f"ItemType(name='{self._item_name}', value={self._value}, weight={self._weight})"


class Item:
    """Represents an instance of an item at a specific location in the world."""

    item_type: ItemType
    location: Location

    def __init__(self, item_type: ItemType, location: Location) -> None:
        self.item_type = item_type
        self.location = location

    def __repr__(self) -> str:
        return f"Item(item_type={self.item_type.__repr__()}, location={self.location.__repr__()})"
