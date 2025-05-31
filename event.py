from typing import Dict, Optional, TYPE_CHECKING
from world import World, Tile

if TYPE_CHECKING:
    from character import Character


class Event:
    def __init__(
        self,
        world: "World",
        tile: "Tile",
        name: Optional[str] = None,
    ) -> None:
        # How do we represent the event data itself?
        #   Entity, Action, Event
        self.world = world
        self.name = name if name else f"Event_{world.get_current_world_time()}"
        self.location = tile.building
        self.world.add_edge(self, "location", self.location)
        self.world.add_edge(self, "occured_on", world.get_current_world_time())

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "location": self.location.name,
            # Add other relevant event attributes if needed
        }


class ChatEvent(Event):
    def __init__(
        self,
        world: "World",
        speaker: "Character",
        message: str,
        name: Optional[str] = None,
    ) -> None:
        tile = speaker.get_current_tile()
        super().__init__(
            world, tile, name if name else f"ChatEvent_{world.get_current_world_time()}"
        )
        self.speaker = speaker
        self.message = message

    def to_dict(self) -> Dict[str, str]:
        event_dict = super().to_dict()
        event_dict.update(
            {
                "speaker_name": self.speaker.name,
                "message": self.message,
                # Add other relevant event attributes if needed
            }
        )
        return event_dict


class HistoricalEvent(Event):
    # TODO: should Event even have a name?
    # Pretty much if an event gets classified as Historical
    # then we need to generate a name
    # Normal Events can eventually become Historical overtime
    #   would need to have a reeval depending on conditions
    #   TODO: drill down on the conditions
    def __init__(self) -> None:
        super().__init__()


# TODO: World has an event log that is easily Searchable by User, Time Period, Location
