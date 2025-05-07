# %%
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from world import World, Building
from retriever import Retriever
from google import genai
from google.genai import types
from typeguard import typechecked
from typing import Dict, Any, Tuple
import ast
from relationships import (
    RelationshipType,
    ActionType,
    HistoryPredicateTypes,
)  # Import the enums
from commons import MODEL_NAME
from declarations import add_edge_declaration, create_character_declaration
from character import Character


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

    def generate_character_story(self) -> str:
        character_prompt = f"""
            <prompt>
            You are a story teller that is describing the four characters in which are going to participate in a Sim-like environement in the same house. 
            Write in the style of a historian writing a history textbook.
            <instructions>
            - For each character:
                - Give a history of their family, including names of their parents and family.
                - Write a short backstory of their life uptil year 0 
                - Write dates in the format of BGB (Before the Game Begins - the start of the game from the perspective of the player)
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
            model=self.model, contents=prompt, config=self.gen_config
        )
        text = """
        The air crackled with an unnatural heat, not the searing blaze of hell, but something more insidious: the stagnant, oily residue of a forgotten war. George Bush, or rather, the essence of him, flickered in the periphery, a phantom limb of a man. His form shimmered, less a person and more a composite of anxieties and accusations. The weights pulsed, a crude metronome counting down to something. > He tried to solidify his form, to anchor himself to a solid reality but the effort felt like trying to grasp smoke. The very definition of him – "war," "hell," "murder" – swirled around him, suffocating, heavy. The ground beneath his spectral feet shifted, the landscape morphing. Initially, the battlefield appeared; a scene of absolute destruction, then an endless desert, the sun a malevolent eye staring from the cloudless sky. Next came a courtroom, the faces of the jury a blur of judgment. Bush tried to speak, to offer some defense, but only a garbled whisper escaped his lips, swallowed by the echoing chambers. Fear, a cold, sharp wire, tightened around his spectral throat. The weight of all he was defined by threatened to crush what remained of him.\n
        """
        triples = self.extractor.extract(text)
        print(triples)

        return text


# %%
