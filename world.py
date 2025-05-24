from enum import Enum
import random
import json
from graph import KnowledgeGraph
from entity import Entity
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from character import Character


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


class Tile(Entity):
    """Represents a single tile within a building."""

    building: "Building"
    local_coords: Tuple[int, int]

    def __init__(
        self,
        building: "Building",
        local_coords: Tuple[int, int],
        world: "World",
        name: Optional[str] = None,
    ) -> None:
        name = name if name else f"{building.name}_tile_{local_coords}"
        super().__init__(name, world)
        self.building = building
        self.local_coords = local_coords
        self.is_occupied = False

        self.world = world
        world.add_edge(self, "part_of", self.building)

    def add_occupant(self, entity: Entity):
        print(
            f"adding {entity.name}",
            self.is_occupied,
            type(entity).__name__ == "Character",
        )
        self.is_occupied = self.is_occupied or type(entity).__name__ == "Character"
        self.world.add_edge(entity, "inside", self.building)
        self.world.add_edge(entity, "occupying", self)
        self.world.add_edge(self, "is_occupied_by", entity)

    def __repr__(self) -> str:
        return f"Tile(name='{self.name}', building='{self.building.name}', coords={self.local_coords}'), is_occupied={self.is_occupied}"

    def to_subject_predicate_object(self) -> List[Tuple[str, str, str]]:
        spo = [
            (self.name, "part_of", self.building.name),
            (self.name, "has_local_coords", str(self.local_coords)),
        ]
        return spo


class Building(Entity):
    """Represents a building."""

    coords: Tuple[int, int]
    province: "Province"
    internal_dims: Tuple[int, int]
    tiles: Dict[Tuple[int, int], "Tile"]
    world: "World"

    def __init__(
        self,
        name: str,
        coords: Tuple[int, int],
        province: "Province",
        world: "World",
        internal_dims: Tuple[int, int] = (3, 3),
    ) -> None:
        """
        Initializes a Building.

        Args:
            name: The name of the building.
            coords: The coordinates of the building within a province.
            province: The province this building belongs to.
            world: The world instance.
            internal_dims: The internal dimensions of the building (width, height).
        """
        super().__init__(name, world)
        self.coords = coords
        self.province = province
        self.world = world
        self.internal_dims = internal_dims
        self.tiles = {}
        self._initialize_tiles()

    def _initialize_tiles(self) -> None:
        """Creates the internal tiles for the building."""
        for x in range(self.internal_dims[0]):
            for y in range(self.internal_dims[1]):
                self.tiles[(x, y)] = Tile(
                    building=self,
                    local_coords=(x, y),
                    world=self.world,
                )

    def get_tile(self, local_coords: Tuple[int, int]) -> Optional["Tile"]:
        """Gets a tile at the given local coordinates within the building."""
        return self.tiles.get(local_coords)

    def find_unoccupied_tile(self) -> Optional["Tile"]:
        """Finds the first unoccupied tile in the building."""
        free_tiles = [tile for tile in self.tiles.values() if not tile.is_occupied]
        return random.choice(free_tiles)

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
        return f"Building(name={self.name}, coords={self.coords}, province={self.province}, internal_dims={self.internal_dims})"


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
        x, y = building_coords
        if not (0 <= x < self.province_dims[0] and 0 <= y < self.province_dims[1]):
            raise ValueError(
                f"Building coordinates ({x}, {y}) are out of the province's bounds {self.province_dims}."
            )
        if self.buildings[x][y] is not None:
            raise ValueError(
                f"Coordinate ({x}, {y}) is already occupied by building '{self.buildings[x][y].name}'."
            )
        building = Building(name, building_coords, self, self.world)

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

    def get_entity(self, name: str) -> List[Entity]:
        return [entity for _, entity in self.graph.out_edges(name)]

    @property
    def locations(self) -> List[Entity]:
        return [node for node in self.graph.nodes if type(node) in [Building]]

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
