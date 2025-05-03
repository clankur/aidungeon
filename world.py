from enum import Enum
import random
import json
from graph import KnowledgeGraph
from entity import Entity
from typing import List, Tuple, Dict
from uuid import UUID


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


class Building(Entity):
    """Represents a building."""

    coords: List[Tuple[int, int]]
    province: "Province"

    def __init__(
        self,
        name: str,
        coords: List[Tuple[int, int]],
        province: "Province",
        world: "World",
    ) -> None:
        """
        Initializes a Building.

        Args:
            name: The name of the building.
            coords: The coordinates the building occupies within a province.
            province: The province this building belongs to.
        """
        super().__init__(name, world)
        self.coords = coords
        self.province = province

    def to_subject_predicate_object(self) -> List[Tuple[str, str, str]]:
        province_spo = self.province.to_subject_predicate_object()
        return province_spo + [
            (self.name, "coords", str(self.coords)),
            (self.name, "inProvince", self.province.name),
        ]

    def __str__(self) -> str:
        province_str = str(self.province).replace("\n", "\n  ")  # Indent province info
        return f"Building(name={self.name}, coords={self.coords}\n province={province_str}\n)"

    def __repr__(self) -> str:
        return f"Building(name={self.name}, coords={self.coords}, province={self.province})"


class Province(Entity):
    """Represents a location using a hierarchical grid system."""

    coords: Tuple[int, int]
    region: "Region"
    buildings: list[list[Building | None]]
    terrain: TerrainType

    def __init__(
        self,
        region: "Region",
        coords: Tuple[int, int],
        province_dims: Tuple[int, int],
        terrain: TerrainType,
        world: "World",
    ) -> None:
        """
        Initializes a Province.

        Args:
            region: The region this province belongs to.
            coords: The coordinates of the province within the region.
            province_dims: The dimensions of the province.
            terrain: The type of terrain at this location.
        """
        super().__init__(name=f"Province_{region.coords}_{coords}", world=world)
        self.buildings = [
            [None for _ in range(province_dims[1])] for _ in range(province_dims[0])
        ]
        self.region = region
        self.coords = coords
        self.terrain = terrain
        self.province_dims = province_dims
        self.world = world

    def create_building(
        self, name: str, building_coords: List[Tuple[int, int]]
    ) -> Building:
        """Adds a building to the province grid."""
        building = Building(name, building_coords, self, self.world)
        for x, y in building.coords:
            if not (0 <= x < self.province_dims[0] and 0 <= y < self.province_dims[1]):
                raise ValueError(
                    f"Building coordinates ({x}, {y}) are out of the province's bounds {self.province_dims}."
                )
            if self.buildings[x][y] is not None:
                raise ValueError(
                    f"Coordinate ({x}, {y}) is already occupied by building '{self.buildings[x][y].name}'."
                )

        for x, y in building.coords:
            self.buildings[x][y] = building
        return building

    def to_subject_predicate_object(self) -> List[Tuple[str, str, str]]:
        region_spo = self.region.to_subject_predicate_object()
        building_spo = []
        return (
            region_spo
            + building_spo
            + [
                (self.name, "coords", str(self.coords)),
                (self.name, "terrain", self.terrain.value),
                (self.name, "dims", str(self.province_dims)),
                (self.name, "inRegion", self.region.name),
            ]
        )

    def __str__(self) -> str:
        region_str = str(self.region).replace("\n", "\n  ")  # Indent region info
        return f"Province(name={self.name}, coords={self.coords}, terrain={self.terrain}\n region={region_str}\n)"

    def __repr__(self) -> str:
        return f"Province(name='{self.name}')"


class Region(Entity):
    """Represents a region of the world."""

    coords: Tuple[int, int]
    dims: Tuple[int, int]
    subregions: list[list[Province]]

    def __init__(
        self,
        coords: Tuple[int, int],
        region_dims: Tuple[int, int],
        province_dims: Tuple[int, int],
        world: "World",
    ) -> None:
        super().__init__(name=f"Region_{coords}", world=world)
        self.coords = coords
        self.region_dims = region_dims
        self.subregions = [
            [
                Province(
                    self,
                    (x, y),
                    province_dims,
                    random.choice(list(TerrainType)),
                    world,
                )
                for y in range(region_dims[1])
            ]
            for x in range(region_dims[0])
        ]

    def to_subject_predicate_object(self) -> List[Tuple[str, str, str]]:
        subregion_spo = []
        return subregion_spo + [
            (self.name, "coords", str(self.coords)),
            (self.name, "dims", str(self.region_dims)),
        ]

    def __repr__(self) -> str:
        return f"Region(name={self.name}, coords={self.coords}, dims={self.region_dims}, n_provinces={self.region_dims[0]*self.region_dims[1]})"

    def __str__(self) -> str:
        return self.__repr__()


class World(KnowledgeGraph):
    """Represents the entire world."""

    curr_time: int = 0
    world_dims: Tuple[int, int]
    regions: dict[Tuple[int, int], Region]

    def __init__(
        self,
        world_dims: Tuple[int, int],
        region_dims: Tuple[int, int],
        province_dims: Tuple[int, int],
        graph: KnowledgeGraph = None,
    ) -> None:
        """
        Initializes a World.

        Args:
            world_dims: The dimensions of the world (length, width).
            region_dims: The dimensions of the regions (length, width).
            province_dims: The dimensions of the subregions (length, width).
            graph: The knowledge graph to use for the world. (default: None)
        """
        self.regions = {}
        self.all_coords = set(
            (x, y) for y in range(world_dims[1]) for x in range(world_dims[0])
        )
        self.world_dims = world_dims
        self.region_dims = region_dims
        self.province_dims = province_dims
        super().__init__(graph)

    def get_region(self, region_coord: Tuple[int, int]) -> Region:
        return self.regions[region_coord]

    def create_region(self, region_coord: Tuple[int, int]) -> Region:
        if region_coord not in self.all_coords:
            raise ValueError(
                f"Coordinates ({region_coord}) are out of bounds for world of dimensions {self.world_dims}"
            )
        if region_coord in self.regions:
            raise ValueError(f"Region at coordinates ({region_coord}) already exists.")
        region = Region(region_coord, self.region_dims, self.province_dims, self)
        self.regions[region_coord] = region
        return region

    def query(self, query: str) -> Dict[UUID, List[Tuple[str, str, str]]]:
        uuid_to_spo = {}
        results = super().query(query)  # may want to modify this as its own function
        for subject, result in results.items():
            uuid_to_spo[subject.uuid] = subject.to_subject_predicate_object()
            for predicate, object in result:
                # TODO: investigate how to handle non-unique objects names
                # We use UUIDs internally but for an LLM UUID's are bunk
                uuid_to_spo[subject.uuid].append((subject.name, predicate, object.name))
        return uuid_to_spo

    def get_current_world_time(self) -> int:
        """Get the current time in the world."""
        return self.curr_time
