# %%
from collections import defaultdict
from typing import List, Tuple
from typeguard import typechecked
import uuid


class KnowledgeGraph:
    def __init__(self) -> None:
        self.graph = {}

    @typechecked
    def add_entity(self, name: str) -> str:
        entity_id = str(uuid.uuid4())
        self.graph[entity_id] = {"name": name}
        return entity_id

    @typechecked
    def add_edge(self, subject_id: str, relationship: str, object_id: str) -> None:
        if subject_id not in self.graph:
            raise ValueError(f"Subject not found in graph: {subject_id}")
        if object_id not in self.graph:
            raise ValueError(f"Object not found in graph: {object_id}")
        self.graph[subject_id][relationship] = object_id

    @typechecked
    @staticmethod
    def build_graph(triples: List[Tuple[str, str, str]]) -> "KnowledgeGraph":
        kg = KnowledgeGraph()
        name_to_id = {}

        for subject, relationship, obj in triples:
            if subject not in name_to_id:
                name_to_id[subject] = kg.add_entity(subject)
            if obj not in name_to_id:
                name_to_id[obj] = kg.add_entity(obj)
            subject_id, object_id = name_to_id[subject], name_to_id[obj]
            kg.add_edge(subject_id, relationship, object_id)
        return kg


# %%


# Q/A
# how do we seperate shared names when adding edges
#   use object ids when adding edges
#   alternatively we can assume names are unique
# how do we handle multiple of something ie).
#   Bob owns multiple dogs Jason and Carl
#   List of dogs = [Jason, Carl]
# when adding edges we can indicate if the edge is Fixed or Infinite/Array or List
#   if over capacity, raise an error
#   ie). can have only 1 (fixed) wife but maybe infinite dogs
# do we want predefine relationships/edges OR support dynamic edges?
#   then we can define relationship constraints to be fixed or infinite

# %%
