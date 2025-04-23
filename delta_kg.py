# %%
import networkx as nx

# %%y
graphml_path = "data/conceptnet-en.graphml"
g: nx.MultiDiGraph = nx.read_graphml(graphml_path)
print(f"Graph loaded with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")
# %%
unique_edge_types = set(data["relation"] for _, _, data in g.edges(data=True))
print(f"Unique edge types: {unique_edge_types}")
# %%
from importlib import reload
import retriever
import graph

# %%
reload(retriever)
reload(graph)
from graph import KnowledgeGraph
from retriever import Retriever

# %%
kg = KnowledgeGraph(g)
# %%
r = Retriever()
# %%
from load_history_book import read_pdf

# %%
new_blurb = read_pdf("data/The-History-Book-DK-2016.pdf", (29, 30))
print(new_blurb)
# %%
relevant_entities = r.get_relevant_entities(new_blurb, kg, create_entities=True)

# %%
# get all edges of relevant entitites (triples)
relevant_triples = []
for entity in relevant_entities:
    results = kg.query(entity)
    triples = [(entity, results[0], results[1]) for results in results]
    relevant_triples.extend(triples)
relevant_triples
# %%
from google import genai
from google.genai import types

# %%
MODEL_NAME = "gemini-2.5-flash-preview-04-17"
client = genai.Client()
prompt = f"""<prompt>
              <instructions>
              From the following text, extract the subject-predicate-object triples for the following entities: {relevant_entities} that would require an update to their state in the Knowledge Graph. 
              The following is the current state of the entities in the Knowledge Graph:
              {relevant_triples}
              Identify any pronouns and replace them with the proper noun they refer to, if it is clearly identifiable within the text that it is referring to a relevant entity.
              Return the extracted triples as a Python list of tuples, where each tuple is in the format: (subject, predicate, object) ensuring there is always a subject, predicate and object.
              Return *only* the raw Python list string, without any markdown formatting (like ```python ... ``` or ``` ... ```), so that it can be directly processed by `ast.literal_eval`.
              </instructions>
              <input_text>{new_blurb}</input_text>
              <output_format_description>Respond only with the Python list of tuples of subject-predicate-object triples. Predicates should be one of the following: {unique_edge_types} </output_format_description>
              </prompt>
          """
# %%
response = client.models.generate_content(
    model=MODEL_NAME,
    contents=prompt,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        temperature=0.0,
    ),
)
print(response.text)

# %%
import ast

# %%
new_triples = ast.literal_eval(response.text)
for triple in new_triples:
    to_node = lambda x: x.lower().replace(" ", "_")
    triple = (to_node(triple[0]), to_node(triple[2]), triple[1])
    kg.graph.add_edge(*triple)
# %%
kg.query("Stone Age")

# %%

# %%
