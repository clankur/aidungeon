# %%
import networkx as nx
from collections import defaultdict
from typing import List, Tuple, Dict, Union
from typeguard import typechecked
from entity import Entity
from uuid import UUID


# %%
class KnowledgeGraph:
    def __init__(self, graph: nx.MultiDiGraph = None) -> None:
        self.graph = graph if graph else nx.MultiDiGraph()

    def query(
        self, query: str
    ) -> Union[List[Tuple[str, Entity]], Dict[Entity, List[Tuple[str, Entity]]]]:
        """Queries the graph for outgoing relationships from a given node."""
        if self.graph.has_node(query):
            # Iterate through outgoing edges (u, v, data)
            # Assumes relationship type is stored in the 'relation' attribute of the edge data
            nodes_with_query = []
            for _, v in self.graph.out_edges(query):
                nodes_with_query.append(v)

            results = {}
            for u in nodes_with_query:
                for _, v, data in self.graph.out_edges(u, data=True):
                    relation = data.get("relation", "related_to")
                    if u not in results:
                        results[u] = []
                    results[u].append((relation, v))
            if len(nodes_with_query) == 1:
                return results[nodes_with_query[0]]
            return results

    def add_node(self, entity: Entity) -> None:
        """Adds a node to the graph."""
        self.graph.add_edge(entity.uuid, entity, relation="id")
        self.graph.add_edge(entity.name, entity, relation="name")

    def add_edge(
        self,
        subject: Union[Entity, str, UUID],
        predicate: str,
        object: Union[Entity, str],
        weight: float = 1.0,
    ) -> None:
        """Adds a weighted edge to the graph."""
        self.graph.add_edge(subject, object, relation=predicate, weight=weight)

    def contains(self, subject: Entity) -> bool:
        """Checks if a node exists in the graph."""
        return self.graph.has_node(subject)


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
