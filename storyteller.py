# %%
import re
import json
from google import genai
from google.genai import types
from typeguard import typechecked
from typing import Dict, Any, List, Tuple
import ast
import einops
import torch
import torch.nn.functional as F

from declarations import add_edge_declaration, create_character_declaration
from commons import MODEL_NAME
from world import World, Building
from retriever import Retriever
from character import Character
from entity import Entity
from event import Event, ChatEvent, SummaryEvent


# %%
def get_list_from_response(response: str) -> list[Any]:
    match = re.search(r"\[.*?\]", response.text, re.DOTALL)
    if match:
        response_text = match.group(0)
        return ast.literal_eval(response_text)
    print("Warning: Could not find a list in the response.")
    try:
        response_text = response.text.strip().strip("```", "```")
        return ast.literal_eval(response_text)
    except (SyntaxError, ValueError):
        print("Error: Failed to parse the response text as a list.")
    return []


# %%
class Extractor:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.extractor = genai.Client()
        self.model_name = model_name
        tools = types.Tool(function_declarations=[add_edge_declaration])
        self.config = types.GenerateContentConfig(
            tools=[tools],
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            seed=0,
            temperature=0,
        )

    def extract(self, text: str) -> list[Dict[str, str]]:
        response = self.extractor.models.generate_content(
            model=self.model_name,
            contents=text,
            config=self.config,
        )
        triples = [
            [p.function_call.args for p in c.content.parts if p.function_call]
            for c in response.candidates
        ]
        return triples


class Storyteller:
    def __init__(self, graph: World) -> None:
        self.graph = graph
        self.retriever = Retriever()
        self.extractor = Extractor()
        self.client = genai.Client()
        self.tools = types.Tool(
            function_declarations=[add_edge_declaration, create_character_declaration]
        )
        self.model_name = MODEL_NAME
        self.gen_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            seed=0,
            temperature=0,
        )
        self.event_log: List["Event"] = []
        self.event_index_by_entity: Dict["Entity", List[int]] = {}

        # Use Gemini embeddings instead of sentence transformers
        self.embedding_model = "gemini-embedding-exp-03-07"

    def encode_text(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
        """
        Encode text using the embedding model with optional normalization.

        Args:
            texts: List of text strings to encode
            normalize: Whether to normalize the embeddings (default: True)

        Returns:
            Tensor of embeddings, optionally normalized
        """
        # Get embeddings from the model
        embeddings = torch.stack(
            [
                self.client.models.embed_content(
                    model=self.embedding_model, content=text
                ).values
                for text in texts
            ]
        )

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def generate_character_story(self) -> str:
        character_prompt = f"""
            <prompt>
            You are a story teller that is describing the four characters in which are going to participate in a Sim-like environement in the same house. 
            Write in the style of a historian writing a history textbook.
            <instructions>
            - For each character:
                - Give a history of their family, including names of their parents and family.
                - Write a short backstory of their life uptil year 0 
                - Write dates in the format of BGB (Before the Game Begins - before the start of the game from the perspective of the player)
                - Describe their physical attributes, including sex, race, birth year
                - Describe their personality
                - Describe their goals and motivations
            </instructions>
            </prompt>
        """

        return self.client.models.generate_content(
            model=MODEL_NAME,
            config=self.gen_config,
            contents=character_prompt,
        ).text

    def init_story(self, characters_story) -> str:
        location: Building = self.graph.get_entity("House")[0]
        # locations = [location.name for location in self.graph.locations]
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(text=characters_story),
                ],
            )
        ]
        create_character_config = types.GenerateContentConfig(
            tools=[self.tools],
            thinking_config=self.gen_config.thinking_config,
            seed=self.gen_config.seed,
            temperature=self.gen_config.temperature,
        )

        response = self.client.models.generate_content(
            model=MODEL_NAME,
            config=create_character_config,
            contents=contents,
        )
        function_calls = [
            [p.function_call for p in c.content.parts if p.function_call]
            for c in response.candidates
        ][0]
        if not function_calls:
            raise ValueError(f"Did not call function {response.text}")
        for call in function_calls:
            if call.name == create_character_declaration.get("name"):
                call.args["world"] = self.graph
                tile = location.find_unoccupied_tile()
                call.args["location"] = tile
                if call.args["location"] is None:
                    raise ValueError(f"No tiles left in {location}")
                character = Character(**call.args)
            else:
                print(call.args)
        return function_calls

    @typechecked
    def generate_next_step(self, query: str) -> str:
        relevant_entities = self.retriever.retrieve(query, self.graph)
        information = "\n".join(
            [f"{v} (weight: {score})" for v, score in relevant_entities]
        )
        prompt = f"""
            You're a dungeon master and storyteller that provides any kind of game, roleplaying and story content.
            Instructions:
            - Be specific, literal, concrete, creative, grounded and clear
            - Avoid reusing themes, sentences, dialog or descriptions
            - Continue unfinished sentences
            - Show realistic consequences

            Use the following information to generate the next step in the story:
            '{information}'
        """
        print(prompt)

        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt, config=self.gen_config
        )
        text = """
        The air crackled with an unnatural heat, not the searing blaze of hell, but something more insidious: the stagnant, oily residue of a forgotten war. George Bush, or rather, the essence of him, flickered in the periphery, a phantom limb of a man. His form shimmered, less a person and more a composite of anxieties and accusations. The weights pulsed, a crude metronome counting down to something. > He tried to solidify his form, to anchor himself to a solid reality but the effort felt like trying to grasp smoke. The very definition of him – "war," "hell," "murder" – swirled around him, suffocating, heavy. The ground beneath his spectral feet shifted, the landscape morphing. Initially, the battlefield appeared; a scene of absolute destruction, then an endless desert, the sun a malevolent eye staring from the cloudless sky. Next came a courtroom, the faces of the jury a blur of judgment. Bush tried to speak, to offer some defense, but only a garbled whisper escaped his lips, swallowed by the echoing chambers. Fear, a cold, sharp wire, tightened around his spectral throat. The weight of all he was defined by threatened to crush what remained of him.\n
        """
        triples = self.extractor.extract(text)
        print(triples)

        return text

    def summarize_history(self, history: List["Event"]) -> List["SummaryEvent"]:
        """
        Create summary events for blocks of events.

        Args:
            history: List of events to summarize
        Returns:
            List of SummaryEvent objects
        """
        if not history:
            return []
        block_size = 5
        summary_events = []
        # Split events into blocks of block_size
        for i in range(0, len(history), block_size):
            event_block: List["Event"] = history[i : i + block_size]

            # Use the tile from the first event that has a valid location
            # TODO: determine if we want to select the most common tile in the block
            representative_tile = None
            for event in event_block:
                representative_tile = event.tile
                break

            # If no representative tile found, skip this block
            if representative_tile is None:
                continue
            # Create SummaryEvent for this block
            summary_name = (
                f"Summary_{self.graph.get_current_world_time()}_{i//block_size}"
            )
            summary_event = SummaryEvent(
                world=self.graph,
                tile=representative_tile,
                event_block=event_block,
                name=summary_name,
            )
            summary_events.append(summary_event)

        return summary_events

    def _calculate_similarity_scores(
        self,
        summary_events: List["SummaryEvent"],
        latest_events: List["Event"],
    ) -> List[Tuple["SummaryEvent", float]]:
        """
        Calculate cosine similarity between summary events and latest events.
        Following the pattern from Retriever class using embeddings.

        Args:
            summary_events: List of SummaryEvent objects
            latest_events: List of recent Event objects
        Returns:
            List of tuples containing (SummaryEvent, similarity_score) sorted by score descending
        """
        if not summary_events or not latest_events:
            return []

        latest_events_text = [json.dumps(event.to_dict()) for event in latest_events]
        summary_texts = [event.summary for event in summary_events]

        # Use the new encode_text function with normalization
        latest_embeddings = self.encode_text(latest_events_text, normalize=True)
        summary_embeddings = self.encode_text(summary_texts, normalize=True)

        # Compute cosine similarity using einops
        similarities = einops.einsum(
            latest_embeddings,
            summary_embeddings,
            "n_latest C, n_summary C -> n_latest n_summary",
        )

    def _select_top_summary_events(
        self,
        summary_events: List["SummaryEvent"],
        latest_events: List["Event"],
        top_n: int = 3,
    ) -> List["SummaryEvent"]:
        """
        Select the top n most relevant summary events based on cosine similarity.

        Args:
            summary_events: List of SummaryEvent objects
            latest_events: List of recent Event objects
            top_n: Number of top events to select

        Returns:
            List of top n most relevant SummaryEvent objects
        """
        scored_summaries = self._calculate_similarity_scores(
            summary_events, latest_events
        )

        # Return top n summary events
        top_events = [event for event, score in scored_summaries[:top_n]]
        return top_events

    def add_event(self, event: "Event") -> None:
        """
        Add an event to the event log and update entity index.

        Args:
            event: The event to add to tracking structures
        """
        # Add to event log
        event_index = len(self.event_log)
        self.event_log.append(event)

        # Update entity index - track which entities are involved in this event
        involved_entities = self._get_involved_entities(event)

        for entity in involved_entities:
            if entity not in self.event_index_by_entity:
                self.event_index_by_entity[entity] = []
            self.event_index_by_entity[entity].append(event_index)

    def _get_involved_entities(self, event: "Event") -> List["Entity"]:
        """
        Get all entities involved in an event.

        Args:
            event: The event to analyze

        Returns:
            List of entities involved in the event
        """
        involved_entities = []

        # All events involve their location
        if hasattr(event, "location") and event.location:
            involved_entities.append(event.location)

        # ChatEvent involves the speaker
        if isinstance(event, ChatEvent) and hasattr(event, "speaker"):
            involved_entities.append(event.speaker)

        # SummaryEvent involves entities from all events in its block
        if isinstance(event, SummaryEvent) and hasattr(event, "event_block"):
            for block_event in event.event_block:
                involved_entities.extend(self._get_involved_entities(block_event))

        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in involved_entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)

        return unique_entities

    def generate_event_response(
        self, responder: Character, latest_events: List["Event"]
    ) -> Event:
        traits_str = "\n".join(
            [f"* {s} {p} {o}" for s, p, o in responder.to_subject_predicate_object()]
        )

        # pull past events that are relevant
        responders_events = self.event_index_by_entity.get(responder, [])
        print(responders_events)

        # Get the actual events from the indices
        responder_event_history = (
            [self.event_log[i] for i in responders_events] if responders_events else []
        )
        if len(responder_event_history) > 10:
            background_events = self.summarize_history(responder_event_history)

            background_history_str = "\n".join(
                f"* { event.summary }" for event in background_events
            )
        else:
            background_history_str = "\n".join(
                f"* {sub, pred, obj}"
                for event in responder_event_history
                for sub, pred, obj in event.to_subject_predicate_object()
            )

        # WHAT IS A SUMMARY EVENT = block of events with a summary

        # TODO:
        # cosine similarity between latest events and summaries
        # of the summary events find the ones most aligned
        # from the top relevant summary events
        # add their full block to the prompt

        # ALTERNATE: can proompt to distinguish among relevant events
        # TODO: weight recent elements higher

        # Create event history string from latest events
        latest_events_str = "\n".join(
            [
                "\n".join(
                    [
                        f"* {sub, pred, obj}"
                        for sub, pred, obj in event.to_subject_predicate_object()
                    ]
                )
                for event in latest_events
            ]
        )
        # Extract
        # speaker, location, message
        # f"{speaker} said {message} in {location}"
        print(f"{latest_events_str}")

        print(f"{background_history_str}")

        # Combine latest events with relevant historical context
        prompt = f"""
            # Background 
            ## Character 
            You are {responder.name} and these are your traits: \n {traits_str}
            ## History
            This is your history of events you recall and remember: \n {background_history_str}
            # Instructions
            What do you say in response to the following events that just occured:\n {latest_events_str}
        """
        print(f"prompt \n { prompt }")
        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt, config=self.gen_config
        ).text

        # Create the chat event and add it to tracking structures
        chat_event = ChatEvent(self.graph, responder, response)
        self.add_event(chat_event)

        return chat_event


# %%
