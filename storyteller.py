# %%
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from graph import KnowledgeGraph
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

MODEL_NAME = "gemini-2.5-flash-preview-04-17"


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

    def extract(self, text: str) -> list[Dict[str, str]]:
        # predicates = the enums from relationships.py
        predicates = (
            [item.value for item in RelationshipType]
            + [item.value for item in ActionType]
            + [item.value for item in HistoryPredicateTypes]
        )

        prompt = f"""<prompt>
            <instructions>
            From the following text, extract the most significant subject-predicate-object triples. 
            Ignore relationships where the subject or object is a general term like "people," "person," "family," "nobles," "parents," "father," "marriage," etc., unless a specific named entity is clearly associated with that term in the text. 
            Focus on extracting triples where specific individuals, families, or named entities are the subject and object.
            Identify any pronouns and replace them with the proper noun they refer to, if it is clearly identifiable within the text.
            Return the extracted triples as a Python list of tuples, where each tuple is in the format: (subject, predicate, object) ensuring there is always a subject, predicate and object.
            Return *only* the raw Python list string, without any markdown formatting (like ```python ... ``` or ``` ... ```), so that it can be directly processed by `ast.literal_eval`.
            </instructions>
            <examples>
            <example>
            <text>Leonardo da Vinci was a painter, sculptor, architect, inventor and mathematician.</text>
            <response>[('Leonardo da Vinci', 'was', 'painter'),('Leonardo da Vinci', 'was', 'sculptor'), ('Leonardo da Vinci', 'was', 'architect'), ('Leonardo da Vinci', 'was', 'inventor'), ('Leonardo da Vinci', 'was', 'mathematician')]</response>
            </example>
            <example>
            <text>Savonarola attacked corruption, angered the pope, was accused of heresy, and was sentenced to death on 23rd of May 1498.</text>
            <response>[('Savonarola', 'attack', 'corruption'), ('pope', 'has_grudge_against', 'Savonarola'), ('Savonarola', 'declared', 'heretic'), ('Savonarola', 'executed', '5/23/1498')]</response>
            </example>
            <example>
            <text>The Medici family were patrons of artists like Michelangelo.</text>
            <response>[('Medici family', 'were patrons of', 'Michelangelo')]</response>
            </example>
            </examples>
            <input_text>{text}</input_text>
            <output_format_description>Respond only with the Python list of tuples of subject-predicate-object triples.</output_format_description>
            </prompt>
        """
        response = self.extractor.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        print(response.text)
        # Use regex to find the list within the response text, handling potential markdown fences
        triples = get_list_from_response(response)
        triples = [triple for triple in triples if triple[1] not in predicates]
        return triples


class Storyteller:
    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph
        self.retriever = Retriever()
        self.extractor = Extractor()
        self.client = genai.Client()
        self.model_name = MODEL_NAME

    def init_story(self) -> Tuple[str, str]:
        world_prompt = f"""
            <prompt>
            You're a dungeon master and storyteller, generating initial the game world but write in the perspective of a historical account you would write in a textbook.
            <instructions>
            - Give a history of the town including important events and figures
            - Write dates in the format of BGB (Before the Game Begins - the start of the game from the perspective of the player)                
            - Describe the town, in which region it is located and their geographical location in the world
            - Describe all the buildings in the town at the time of 0 BGB 
            - Describe the historical events and figures in the town and the region/province leading up to 0 BGB
            </instructions>
            </prompt>
        """

        world_response = self.client.models.generate_content(
            model=self.model_name,
            contents=world_prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

        character_prompt = f"""
            <prompt>
            You're a dungeon master and storyteller, generating initial the game world but write in the perspective of a historical account you would write in a textbook.
            <game_world>{world_response.text}</game_world>
            <instructions>
            - For each character:
                - Give a history of their family 
                - Write a short backstory of their life uptil year 0 BGB (Before the Game Begins - the start of the game from the perspective of the player)
                - Write dates in the format of BGB                 
                - Describe their physical attributes, including sex, race, birth year
                - Describe their personality
                - Describe their goals and motivations
            </instructions>
            </prompt>
        """
        character_response = self.client.models.generate_content(
            model=self.model_name,
            contents=character_prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        # prompt it to create JSON objects?
        # or create character using tool use?

        # world_triples = self.extractor.extract(world_response.text)
        # character_triples = self.extractor.extract(character_response.text)
        return world_response.text, character_response.text

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
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=512,
            ),
        )
        text = """
        The air crackled with an unnatural heat, not the searing blaze of hell, but something more insidious: the stagnant, oily residue of a forgotten war. George Bush, or rather, the essence of him, flickered in the periphery, a phantom limb of a man. His form shimmered, less a person and more a composite of anxieties and accusations. The weights pulsed, a crude metronome counting down to something. > He tried to solidify his form, to anchor himself to a solid reality but the effort felt like trying to grasp smoke. The very definition of him – "war," "hell," "murder" – swirled around him, suffocating, heavy. The ground beneath his spectral feet shifted, the landscape morphing. Initially, the battlefield appeared; a scene of absolute destruction, then an endless desert, the sun a malevolent eye staring from the cloudless sky. Next came a courtroom, the faces of the jury a blur of judgment. Bush tried to speak, to offer some defense, but only a garbled whisper escaped his lips, swallowed by the echoing chambers. Fear, a cold, sharp wire, tightened around his spectral throat. The weight of all he was defined by threatened to crush what remained of him.\n
        """
        triples = self.extractor.extract(text)
        print(triples)

        return text


# %%
if __name__ == "__main__":
    from load_conceptnet import load_conceptnet_csv

    # %%
    en_conceptnet_path = "data/conceptnet-assertions-5.7.0-en.csv"
    graph = KnowledgeGraph(load_conceptnet_csv(en_conceptnet_path))
    # %%
    storyteller = Storyteller(graph)
    output = storyteller.generate_next_step("Where is President George Bush?")
    print(output)
# %%
