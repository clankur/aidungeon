# %%
import json
from grounded_dungeon import Retriever
from graph import KnowledgeGraph

# %%
wiki_data = json.load(open("re-nlg_0-10000.json", "r"))
triples = set()
for obj in wiki_data:
    for triple in obj["triples"]:
        sub, pred, obj = triple["subject"], triple["predicate"], triple["object"]
        sub, pred, obj = sub["surfaceform"], pred["surfaceform"], obj["surfaceform"]
        if sub and pred and obj:
            triples.add((sub, pred, obj))

triples = list(triples)
# %%
graph = KnowledgeGraph.build_graph(triples)

# %%
graph.graph

# %%
retriever = Retriever(graph)

# %%
retriever.retrieve("Who taught Aristotle?")

# %%
retriever.retrieve("Who are the students of Plato?")

# %%
graph.graph.keys()

# %%
