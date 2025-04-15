# %%
from rdflib import Graph, URIRef
from collections import defaultdict
from typing import List, Tuple
from typeguard import typechecked
import uuid


# %%
class KnowledgeGraph:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def query(self, query: str) -> List[str]:
        query = query.replace(" ", "_").lower()
        query_uri = URIRef(f"/c/en/{query}")
        query_string_props = f"""
            SELECT ?p ?o
            WHERE {{
            <{query_uri}> ?p ?o .
        }}
        """
        return self.graph.query(query_string_props)

    def contains(self, query: str) -> bool:
        query = query.replace(" ", "_").lower()
        query_uri = URIRef(f"/c/en/{query}")
        ask_query = f"ASK {{ <{query_uri}> ?p ?o . }}"
        results = self.graph.query(ask_query)
        return bool(results)


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

# do we need to build seperate schemas for each entity type?
#   ie). human has a different schema from pet, location, object, etc
#   how granular do we want our schemas?
#       pet and human are similar enough that we can treat them the same
# %%
