# %%
import networkx as nx
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Optional
from typeguard import typechecked
from entity import Entity
from uuid import UUID


# %%
class KnowledgeGraph:
    def __init__(self, graph: nx.MultiDiGraph = None) -> None:
        self.graph = graph if graph else nx.MultiDiGraph()

    def query(self, query: str) -> Dict[Entity, List[Tuple[str, Entity]]]:
        """Queries the graph for outgoing relationships from a given node."""
        if self.graph.has_node(query):
            # Iterate through outgoing edges (u, v, data)
            # Assumes relationship type is stored in the 'relation' attribute of the edge data
            nodes_with_query = []
            for _, v in self.graph.out_edges(query):
                nodes_with_query.append(v)
            results = {}
            for u in nodes_with_query:
                if u not in results:
                    results[u] = []
                for _, v, data in self.graph.out_edges(u, data=True):
                    relation = data.get("relation", "related_to")
                    results[u].append((relation, v))
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

    def get_edges(
        self,
        source: Optional[Union[Entity, str, UUID]] = None,
        predicate: Optional[str] = None,
        object_node: Optional[Union[Entity, str, UUID]] = None,
    ) -> List[Tuple[Union[Entity, str, UUID], str, Union[Entity, str, UUID]]]:
        """
        Retrieves edges from the graph based on source, predicate, and/or object.

        Args:
            source: The source node of the edge.
            predicate: The type of relationship (edge's 'relation' attribute).
            object_node: The target node of the edge.

        Returns:
            A list of tuples, where each tuple is (source, predicate, object_node).
        """
        edges_found = []

        candidate_edges_iter = None

        if source is not None:
            if self.graph.has_node(source):
                # Get all outgoing edges from the source
                # For MultiDiGraph, out_edges(u, data=True) gives (u, v, data) for all v
                candidate_edges_iter = self.graph.out_edges(source, data=True)
            else:  # Source node doesn't exist, so no edges from it
                return []
        else:
            # Iterate through all edges if no source is specified
            candidate_edges_iter = self.graph.edges(data=True)

        for u, v, data in candidate_edges_iter:
            edge_predicate = data.get("relation")

            # If source was specified, u will always match source due to how candidate_edges_iter was formed.
            # So we only need to check predicate and object_node if they are specified.

            predicate_match = (predicate is None) or (edge_predicate == predicate)
            object_match = (object_node is None) or (v == object_node)

            if predicate_match and object_match:
                edges_found.append((u, edge_predicate, v))

        return edges_found

    def remove_edge(
        self,
        subject: Union[Entity, str, UUID],
        predicate: str,
        object_node: Union[Entity, str, UUID],
    ) -> None:
        """
        Removes an edge from the graph based on source, predicate, and object.

        Args:
            subject: The source node of the edge.
            predicate: The type of relationship (edge's 'relation' attribute).
            object_node: The target node of the edge.
        """
        # Find the edge keys with matching relation
        if not self.graph.has_node(subject) or not self.graph.has_node(object_node):
            return

        edge_keys_to_remove = []
        # Get all edges between subject and object_node
        if self.graph.has_edge(subject, object_node):
            for key, data in self.graph.get_edge_data(subject, object_node).items():
                if data.get("relation") == predicate:
                    edge_keys_to_remove.append(key)

        # Remove the edges using the keys
        for key in edge_keys_to_remove:
            self.graph.remove_edge(subject, object_node, key)


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
