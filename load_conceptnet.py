# %%
from rdflib import Graph, URIRef
import csv
import os


# %%
def load_conceptnet_csv(file_path: str) -> Graph:
    """
    Loads ConceptNet edges from a CSV file into an rdflib Graph,
    filtering for edges where both subject and object nodes belong to the specified language.

    The CSV file format is expected to be tab-separated with five fields per line:
    edge_uri, relation, start_node, end_node, metadata_json

    Example line:
    /a/[/r/Antonym/,/c/ab/агыруа/n/,/c/ab/аҧсуа/]   /r/Antonym      /c/ab/агыруа/n  /c/ab/аҧсуа     {"dataset": "/d/wiktionary/en", ...}

    Args:
        file_path: Path to the ConceptNet CSV file (can be gzipped).
        language_code: The 2-letter language code (e.g., "en") to filter nodes by.
                       Only edges where both start and end nodes match `/c/{language_code}/`
                       will be included.

    Returns:
        An rdflib Graph containing the filtered triples.
    """
    g = Graph()
    nodes_added = 0
    lines_processed = 0

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

                    # Convert ConceptNet URIs to rdflib URIRefs
                    subject = URIRef(start)
                    predicate = URIRef(rel)
                    obj = URIRef(end)
                    g.add((subject, predicate, obj))
                    nodes_added += 1

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")

    print(
        f"Finished loading. Processed {lines_processed} lines, added {nodes_added} triples."
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
    g = load_conceptnet_csv(en_conceptnet_path)
    print(f"Total triples in graph: {len(g)}")
    # %%
    from importlib import reload
    import retriever

    # %%
    reload(retriever)
    from retriever import Retriever, KnowledgeGraph

    # %%
    r = Retriever(g)
    # %%
    r.retrieve("What is the capital of the United States?")

    # %%
    r.retrieve("Where is George Bush?")
    # %%
    r.retrieve("Where is Mount Everest?")
    # %%
    r.retrieve("What does Apple do?")
    # %%
    r.retrieve("What does Windows do?")
    # %%
    r.retrieve("Should I add Windows to my house?")
    # %%
    r.retrieve("What is a Macintosh?")

# %%
