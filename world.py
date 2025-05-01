from enum import Enum
import random
import json
from graph import KnowledgeGraph
from entity import Entity
from typing import List, Tuple


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


class Subregion(Entity):
    """Represents a location using a hierarchical grid system."""

    subregion_coords: Tuple[int, int]
    terrain: TerrainType
    region: "Region"
    # buildings: list[Building] TODO

    def __init__(
        self,
        region: "Region",
        subregion_coords: Tuple[int, int],
        terrain: TerrainType,
    ) -> None:
        """
        Initializes a Location.

        Args:
            region: The region this location belongs to.
            subregion_x: The x-coordinate within the region's subgrid.
            subregion_y: The y-coordinate within the region's subgrid.
            terrain: The type of terrain at this location.
        """
        # Need to call Entity's __init__
        super().__init__(
            name=f"Subregion_{region.coords}_{subregion_coords}"
        )  # Provide a name to Entity
        self.region = region
        self.subregion_coords = subregion_coords
        self.terrain = terrain

    def __repr__(self) -> str:
        region_info = {
            "dims": self.region.dims,
        }

        data = {
            "region": region_info,
            "subregion_coords": self.subregion_coords,
            "terrain": (
                self.terrain.value
                if hasattr(self.terrain, "value")
                else str(self.terrain)
            ),  # Get enum value or string
        }
        return json.dumps(data)


class Region:
    """Represents a region of the world."""

    coords: Tuple[int, int]
    dims: Tuple[int, int]
    subregions: list[list[Subregion]]

    def __init__(self, coords: Tuple[int, int], dims: Tuple[int, int]) -> None:
        self.coords = coords
        self.dims = dims
        self.subregions = [
            [
                Subregion(self, (x, y), random.choice(list(TerrainType)))
                for x in range(dims[0])
            ]
            for y in range(dims[1])
        ]

    def __repr__(self) -> str:
        return f"Region(dims={self.dims}, num_locations={self.dims[0]*self.dims[1]})"


class World(KnowledgeGraph):
    """Represents the entire world."""

    curr_time: int = 0
    dims: Tuple[int, int]
    regions: dict[Tuple[int, int], Region]

    def __init__(
        self,
        dims: Tuple[int, int],
        region_dims: Tuple[int, int],
        graph: KnowledgeGraph = None,
    ) -> None:
        self.dims = dims
        self.regions = {}
        self.all_coords = set((x, y) for x in range(dims[0]) for y in range(dims[1]))
        self.region_dims = region_dims
        super().__init__(graph)

    def get_region(self, x: int, y: int) -> Region:
        return self.regions[(x, y)]

    def create_region(self, x: int, y: int) -> Region:
        if (x, y) not in self.all_coords:
            raise ValueError(
                f"Coordinates ({x}, {y}) are out of bounds for world of dimensions {self.dims}"
            )
        region = Region((x, y), self.region_dims)
        self.regions[(x, y)] = region
        return region

    def get_current_world_time(self) -> int:
        """Get the current time in the world."""
        return self.curr_time
