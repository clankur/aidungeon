# %%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from graph import KnowledgeGraph
from retriever import Retriever
from google import genai
from google.genai import types
from typeguard import typechecked
from typing import Dict


# %%
class Extractor:
    def __init__(self) -> None:
        self.triplet_extractor = pipeline(
            "text2text-generation",
            model="Babelscape/rebel-large",
            tokenizer="Babelscape/rebel-large",
        )

    def extract(self, text: str) -> list[Dict[str, str]]:
        text = self.triplet_extractor.tokenizer.batch_decode(
            [
                self.triplet_extractor(
                    text,
                    return_tensors=True,
                    return_text=False,
                )[
                    0
                ]["generated_token_ids"]
            ]
        )[0]

        triplets = []
        relation, subject, relation, object_ = "", "", "", ""
        text = text.strip()
        current = "x"
        for token in (
            text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split()
        ):
            if token == "<triplet>":
                current = "t"
                if relation != "":
                    triplets.append(
                        {
                            "head": subject.strip(),
                            "type": relation.strip(),
                            "tail": object_.strip(),
                        }
                    )
                    relation = ""
                subject = ""
            elif token == "<subj>":
                current = "s"
                if relation != "":
                    triplets.append(
                        {
                            "head": subject.strip(),
                            "type": relation.strip(),
                            "tail": object_.strip(),
                        }
                    )
                object_ = ""
            elif token == "<obj>":
                current = "o"
                relation = ""
            else:
                if current == "t":
                    subject += " " + token
                elif current == "s":
                    object_ += " " + token
                elif current == "o":
                    relation += " " + token
        if subject != "" and relation != "" and object_ != "":
            triplets.append(
                {
                    "head": subject.strip(),
                    "type": relation.strip(),
                    "tail": object_.strip(),
                }
            )
        return triplets


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
        # convert response.text into a new subject relationship object in graph

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
