# %%
from rdflib import Graph, term, URIRef, Literal
import csv
import random
from typing import List


# %%
def load_conceptnet_csv(file_path: str, language_code: str = "en") -> Graph:
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
                lines_processed = i + 1
                if lines_processed % 5000000 == 0:  # Print progress less often
                    print(
                        f"Processed {lines_processed // 1000000}M lines, added {nodes_added} '{language_code}' triples so far..."
                    )

                if len(row) == 5:
                    # Fields: edge_uri, relation, start_node, end_node, metadata_json
                    _, rel, start, end, _ = row

                    # --- Language Filter Check ---
                    if not (
                        start.startswith(lang_prefix) and end.startswith(lang_prefix)
                    ):
                        continue  # Skip if either node is not in the target language

                    try:
                        # Convert ConceptNet URIs to rdflib URIRefs
                        subject = URIRef(start)
                        predicate = URIRef(rel)
                        obj = URIRef(end)
                        g.add((subject, predicate, obj))
                        nodes_added += 1
                    except Exception as e:
                        # Print errors less often to avoid spamming console on large files
                        if random.random() < 0.001:
                            print(
                                f"Skipping row {i+1} due to parsing error: {e} | Row: {row}"
                            )
                else:
                    # Print errors less often
                    if random.random() < 0.001:
                        print(
                            f"Skipping malformed row {i+1}: Expected 5 fields, got {len(row)} | Row: {row}"
                        )

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print(
        f"Finished loading. Processed {lines_processed} lines, added {nodes_added} triples for language '{language_code}'."
    )
    return g


# Example usage (optional, requires a conceptnet csv file)
# if __name__ == "__main__":
#     # Replace with the actual path to your conceptnet file
#     conceptnet_file = "path/to/conceptnet-assertions-5.7.0.csv.gz"
#     print(f"Loading ConceptNet data from {conceptnet_file}...")
#     graph = load_conceptnet_csv(conceptnet_file)
#     print(f"Loaded {len(graph)} triples.")
#
#     # Example query: Find relations for the concept "cat"
#     cat_uri = URIRef("/c/en/cat")
#     print(f"\nTriples involving {cat_uri}:")
#     for s, p, o in graph.triples((cat_uri, None, None)):
#         print(f"  {s} {p} {o}")
#     for s, p, o in graph.triples((None, None, cat_uri)):
#         print(f"  {s} {p} {o}")

# %%
# Load only English-to-English edges
target_language = "en"
print(f"Loading graph with only '{target_language}' language edges...")
g = load_conceptnet_csv(
    "data/conceptnet-assertions-5.5.0.csv.gz", language_code=target_language
)
print(f"Total triples in '{target_language}'-only graph: {len(g)}")
# %%
g.all_nodes()

# %%
query_uri = term.URIRef("/c/en/karate/n")
query_string_props = f"""
        SELECT ?p ?o
        WHERE {{
        <{query_uri}> ?p ?o .
    }}
"""
results_props = g.query(query_string_props)
for row in results_props:
    print(row)
# %%
