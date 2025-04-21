# %%
import networkx as nx
from collections import defaultdict
from typing import List, Tuple
from typeguard import typechecked
import uuid


# %%
class KnowledgeGraph:
    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self.graph = graph

    def query(self, query: str) -> List[Tuple[str, str]]:
        """Queries the graph for outgoing relationships from a given node."""
        node_name = query.replace(" ", "_").lower()
        results = []
        if self.graph.has_node(node_name):
            # Iterate through outgoing edges (u, v, data)
            # Assumes relationship type is stored in the 'relation' attribute of the edge data
            for _u, v, data in self.graph.out_edges(node_name, data=True):
                relation = data.get(
                    "relation", "related_to"
                )  # Default if 'relation' attr is missing
                results.append((relation, v))
        return results

    def contains(self, query: str) -> bool:
        """Checks if a node exists in the graph."""
        node_name = query.replace(" ", "_").lower()
        print(f"{node_name=}")
        # Use networkx's has_node method
        return self.graph.has_node(node_name)


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
