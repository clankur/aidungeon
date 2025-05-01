from enum import Enum
from world import Subregion, Entity
import uuid
from uuid import UUID


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


class Item(Entity):
    """Represents an instance of an item at a specific location in the world."""

    item_type: ItemType
    location: "Location"

    def __init__(self, item_type: ItemType, location: "Location") -> None:
        super().__init__(item_type._item_name)
        self.item_type = item_type
        self.location = location

    def __repr__(self) -> str:
        location_repr = str(self.location)
        try:
            location_repr = self.location.__repr__()
        except AttributeError:
            pass
        return f"Item(item_type={self.item_type.__repr__()}, location={location_repr})"
