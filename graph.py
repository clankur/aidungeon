# %%
from collections import defaultdict
from typing import List, Tuple
from typeguard import typechecked
import uuid


class KnowledgeGraph:
    def __init__(self) -> None:
        self.graph = {}

    @typechecked
    def add_entity(self, name: str) -> None:
        self.graph[name] = {"name": name}

    @typechecked
    def add_edge(self, subject: str, relationship: str, object: str) -> None:
        if subject not in self.graph:
            self.add_entity(subject)
        if object not in self.graph:
            self.add_entity(object)
        self.graph[subject][relationship] = object

    @typechecked
    @staticmethod
    def build_graph(triples: List[Tuple[str, str, str]]) -> "KnowledgeGraph":
        kg = KnowledgeGraph()
        for subject, relationship, obj in triples:
            kg.add_edge(subject, relationship, obj)
        return kg


# %%


# Q/A
# how do we seperate shared entity names
#   Solution: use object ids when adding edges
#   Solution: alternatively we can assume names are unique [X]

# how do we handle bidirectional relationships?
#   Bidrectional: Bob is married to Alice so Alice is married to Bob
#   Not bidirectional: Bob owns Jason so Jason's owner is Bob
#       treat owns as a seperate edge as owner

# how do we handle multiple of something ie).
#   Bob owns multiple dogs Jason and Carl
#   List of dogs = [Jason, Carl]
#   Solution: when adding edges we can indicate if the edge is Fixed or Infinite/Array or List
#       if over capacity, raise an error
#       ie). can have only 1 (fixed) wife but maybe infinite dogs

# do we want predefine relationships/edges OR support dynamic edges?
#   ie). do we have fixed set of edges/relationships an object in the graph can have?
#   simpler and more structured if we predefine
#   Entity's can have these set of edges {
#       owns: [List]
#       married: [1 element]
#       on_head: [1 element]
#       on_torso:
#       on_waste:
#       on_back:
#       in_left_hand:
#       in_right_hand:
#       location: (?)
#   }
# %%
