# %%
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from graph import KnowledgeGraph
from retriever import Retriever
from google import genai
from google.genai import types
from typeguard import typechecked
from typing import Dict
import ast


# %%
class Extractor:
    def __init__(self) -> None:
        self.extractor = genai.Client()

    def extract(self, text: str) -> list[Dict[str, str]]:
        prompt = f"""
            From the following text, extract all subject-predicate-object triples.
            Identify any pronouns and replace them with the proper noun they refer to, if it is clearly identifiable within the text.
            Return the extracted triples as a Python list of tuples, where each tuple is in the format: (subject, predicate, object) 
            ensuring there is Subject, Predicate and Object. Return only the list so that it can be processed by ast.literal_eval with no markdown formatting.

            Example 1:
            Text: "Alice loves Bob. She admires him greatly."
            Response: [("Alice", "loves", "Bob"), ("Alice", "admires greatly", "Bob")]

            Example 2:
            Text: "The cat sat on the mat. It looked comfortable."
            Response: [("cat", "sat on", "mat"), ("cat", "looked comfortable", None)] 

            Example 3:
            Text: "John went to the store. He bought milk."
            Response: [("John", "went to", "store"), ("John", "bought", "milk")]

            Text: '{text}'
            Response:
        """

        response = self.extractor.models.generate_content(
            model="gemini-2.0-flash-001", contents=prompt
        )
        print(response.text)
        # Use regex to find the list within the response text, handling potential markdown fences
        match = re.search(r"\[.*?\]", response.text, re.DOTALL)
        if match:
            response_text = match.group(0)
            return ast.literal_eval(response_text)
        else:
            # Handle cases where the list format might be unexpected or not found
            print("Warning: Could not find a list in the response.")
            # Attempt to strip and parse anyway, or return an empty list/raise error
            try:
                response_text = response.text.strip().strip("```", "```")
                return ast.literal_eval(response_text)
            except (SyntaxError, ValueError):
                print("Error: Failed to parse the response text as a list.")
                return []  # Return an empty list as a fallback


class Storyteller:
    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph
        self.retriever = Retriever()
        self.extractor = Extractor()
        self.client = genai.Client()
        self.model = "gemini-2.0-flash-lite-001"

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
            - Continue the text where it ends without repeating
            - Avoid reusing themes, sentences, dialog or descriptions
            - Continue unfinished sentences
            - > means an action attempt; it is forbidden to output >
            - Show realistic consequences

            Use the following information to generate the next step in the story:
            '{information}'
        """
        print(prompt)

        # response = self.client.models.generate_content(
        #     model=self.model,
        #     contents=prompt,
        #     config=types.GenerateContentConfig(
        #         max_output_tokens=512,
        #     ),
        # )
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
