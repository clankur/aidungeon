# %%
import csv
import os
import networkx as nx
import re
from typing import Tuple

RELATIONSHIP_MAP = {
    "Antonym": "is the opposite of",
    "AtLocation": "is located at",
    "CapableOf": "is capable of",
    "Causes": "causes",
    "CausesDesire": "makes someone want to",
    "CreatedBy": "is created by",
    "DefinedAs": "is defined as",
    "DerivedFrom": "is derived from",
    "Desires": "wants",
    "DistinctFrom": "is distinct from",
    "Entails": "entails",
    "EtymologicallyDerivedFrom": "comes etymologically from",
    "EtymologicallyRelatedTo": "is etymologically related to",
    "FormOf": "is a form of",
    "HasA": "has a",
    "HasContext": "is used in the context of",
    "HasProperty": "has the property of",
    "InstanceOf": "is an instance of",
    "IsA": "is a",
    "LocatedNear": "is located near",
    "MadeOf": "is made of",
    "MannerOf": "is a manner of",
    "MotivatedByGoal": "is motivated by the goal of",
    "NotCapableOf": "is not capable of",
    "NotDesires": "does not desire",
    "NotHasProperty": "does not have the property of",
    "PartOf": "is part of",
    "RelatedTo": "is related to",
    "SimilarTo": "is similar to",
    "SymbolOf": "is a symbol of",
    "Synonym": "is a synonym of",
    "UsedFor": "is used for",
}


# Helper function to extract node/relation name from ConceptNet URI
def _extract_name_from_uri(uri: str) -> str:
    """Extracts the name part from a ConceptNet URI like /c/en/my_node -> my_node"""
    match = re.match(r"/[rc]/[^/]+/([^/]+)", uri)
    if match:
        return match.group(1)
    return uri


# %%
def load_conceptnet_csv(file_path: str) -> nx.MultiDiGraph:
    """
    Loads ConceptNet edges from a CSV file into a networkx MultiDiGraph.

    The CSV file format is expected to be tab-separated with five fields per line:
    edge_uri, relation, start_node, end_node, metadata_json

    Nodes are represented as strings (extracted from ConceptNet URIs),
    and edges are added with the relationship type stored in the 'relation' attribute.

    Args:
        file_path: Path to the ConceptNet CSV file (can be gzipped).

    Returns:
        A networkx MultiDiGraph containing the ConceptNet data.
    """
    g = nx.MultiDiGraph()

    try:
        # Handle potential gzipped files transparently
        open_func = open
        if file_path.endswith(".gz"):
            import gzip

            open_func = gzip.open

        with open_func(file_path, "rt", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for i, row in enumerate(reader):
                lines_processed = i + 1
                if len(row) == 5:
                    # Fields: edge_uri, relation, start_node, end_node, metadata_json
                    _, rel, start, end, _ = row

                    # Extract clean names for nodes and relation
                    subject_name = _extract_name_from_uri(start)
                    relation_name = _extract_name_from_uri(rel)
                    object_name = _extract_name_from_uri(end)

                    # Add edge to the MultiDiGraph
                    # Nodes are added implicitly if they don't exist
                    g.add_edge(subject_name, object_name, relation=relation_name)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")

    print(
        f"Finished loading. Processed {lines_processed} lines, added {g.number_of_edges()} edges."
    )
    return g


def clean_conceptnet_graph(
    file_path: str, output_path: str, language_code: str = "en"
) -> None:
    """
    Loads ConceptNet graph, removes all nodes that are not in the specified language,
    and saves the cleaned graph to a new file.

    Args:
        file_path: Path to the ConceptNet CSV file (can be gzipped).
        language_code: The 2-letter language code (e.g., "en") to filter nodes by.
                       Only edges where both start and end nodes match `/c/{language_code}/`
                       will be included.
    """
    outfile = open(output_path, "w")
    lang_prefix = f"/c/{language_code}/"
    try:
        # Handle potential gzipped files transparently
        open_func = open
        if file_path.endswith(".gz"):
            import gzip

            open_func = gzip.open

        with open_func(file_path, "rt", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            print(f"Starting load, filtering for language: '{language_code}'...")
            for i, row in enumerate(reader):
                if len(row) == 5:
                    _, rel, start, end, _ = row
                    if start.startswith(lang_prefix) and end.startswith(lang_prefix):
                        outfile.write("\t".join(row) + "\n")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    outfile.close()


def get_en_pred_obj(pred_obj: Tuple[str, str]) -> str:
    """
    converts a tuple of (pred_obj, score) to a string of the form. ie).
    (/r/IsA, /c/en/president_of_united_states) -> "IsA president of united states"
    """
    pred_uri, obj_uri = pred_obj

    pred_name = str(pred_uri).split("/")[-1]
    pred_name = RELATIONSHIP_MAP.get(pred_name, pred_name)
    obj_name = str(obj_uri).replace("/c/en/", "").replace("_", " ")

    return f"{pred_name} {obj_name}"


# %%
if __name__ == "__main__":
    conceptnet_url = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
    conceptnet_path = "data/conceptnet-assertions-5.7.0.csv.gz"
    en_conceptnet_path = "data/conceptnet-assertions-5.7.0-en.csv"
    # %%
    if not os.path.exists(conceptnet_path):
        print(f"Downloading ConceptNet from {conceptnet_url}...")
        os.system(f"wget {conceptnet_url} -O {conceptnet_path}")

    if not os.path.exists(en_conceptnet_path):
        clean_conceptnet_graph(
            conceptnet_path,
            en_conceptnet_path,
            language_code="en",
        )

    # %%
    # Load only English-to-English edges
    graphml_path = "data/conceptnet-en.graphml"
    if not os.path.exists(graphml_path):
        print(f"Loading ConceptNet from {en_conceptnet_path}...")
        g = load_conceptnet_csv(en_conceptnet_path)
        try:
            nx.write_graphml(g, graphml_path)
            print(f"Graph saved to {graphml_path}")
        except Exception as e:
            print(f"Error saving graph to GraphML: {e}")
    else:
        print(f"Loading ConceptNet from {graphml_path}...")
        g = nx.read_graphml(graphml_path)
    print(
        f"Graph loaded with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges."
    )
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
    r = Retriever()
    # %%
    r.retrieve("What is the capital of the United States of America?", kg)

    # %%
    r.retrieve("Where is George Bush?", kg)
    # %%
    r.retrieve("Where is Mount Everest?", kg)
    # %%
    r.retrieve("What does Apple do?", kg)
    # %%
    r.retrieve("What does Windows do?", kg)
    # %%gggggg
    r.retrieve("Should I add Windows to my house?", kg)
    # %%
    r.retrieve("What is a Macintosh?", kg)

# %%
