from typing import List, Dict, Optional, TYPE_CHECKING, Tuple
import json
from google import genai
from google.genai import types
from world import World, Tile
from commons import MODEL_NAME

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
        self.tile = tile
        self.world.add_edge(self, "location", self.location)
        self.world.add_edge(self, "occured_on", world.get_current_world_time())

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "location": self.location.name,
            # Add other relevant event attributes if needed
        }

    def to_subject_predicate_object(self) -> List[Tuple[str, str, str]]:
        """Generate SPO triples for the basic event."""
        return [
            (self.name, "type", "Event"),
            (self.name, "location", self.location.name),
            (self.name, "occurred_on", str(self.world.get_current_world_time())),
        ]


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

    def to_subject_predicate_object(self) -> List[Tuple[str, str, str]]:
        """Generate SPO triples for the chat event."""
        base_spo = super().to_subject_predicate_object()
        chat_spo = [
            (self.speaker.name, "said", self.message),
            (self.speaker.name, "spoke_at", self.location.name),
        ]
        return base_spo + chat_spo


class SummaryEvent(Event):
    def __init__(
        self,
        world: World,
        tile: Tile,
        event_block: List[Event],
        name: str | None = None,
    ) -> None:
        super().__init__(world, tile, name)
        self.event_block = event_block

        event_history_str = "\n".join(
            map(lambda event: json.dumps(event.to_dict()), event_block)
        )

        client = genai.Client()
        prompt = f"""
            <instruction>
            You are an expert summarizer. Generate a concise summary of the following events:
            {event_history_str} 
            </instruction>
        """
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                seed=0,
                temperature=0,
            ),
        )
        # NOTE: this is not a SPO triple as other events have been

        self.summary = response.text  # prompt Gemini to make a summary

        # does the name have the summary of the events
        # lets make a summary just a few sentences (upto 3?)

    def to_subject_predicate_object(self) -> List[Tuple[str, str, str]]:
        """Generate SPO triples for the summary event including the summarized events."""
        spo_triples = []

        # Add basic SPO for the summary event itself
        spo_triples.extend(
            [
                (self.name, "type", "SummaryEvent"),
                (self.name, "summary", self.summary),
                (self.name, "location", self.location.name),
                (self.name, "num_events_summarized", str(len(self.event_block))),
            ]
        )

        # Add SPO triples from all events in the block
        for event in self.event_block:
            if hasattr(event, "to_subject_predicate_object"):
                event_spo = event.to_subject_predicate_object()
                spo_triples.extend(event_spo)
            else:
                # Fallback for events without SPO method
                spo_triples.extend(
                    [
                        (event.name, "type", event.__class__.__name__),
                        (
                            event.name,
                            "location",
                            (
                                event.location.name
                                if hasattr(event, "location")
                                else "unknown"
                            ),
                        ),
                    ]
                )

        return spo_triples

    def to_dict(self) -> Dict[str, str]:
        event_dict = super().to_dict()
        event_dict.update(
            {
                "summary": self.summary,
                "num_events_summarized": str(len(self.event_block)),
                "event_block_names": [event.name for event in self.event_block],
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
