# %%
from rdflib import Graph, term, URIRef
import csv
import random


# %%
def load_conceptnet_csv(file_path: str, sample_rate: float = 1.0) -> Graph:
    """
    Loads ConceptNet edges from a CSV file into an rdflib Graph, with optional sampling.

    The CSV file format is expected to be tab-separated with five fields per line:
    edge_uri, relation, start_node, end_node, metadata_json

    /a/[/r/Antonym/,/c/ab/агыруа/n/,/c/ab/аҧсуа/]   /r/Antonym      /c/ab/агыруа/n  /c/ab/аҧсуа     {"dataset": "/d/wiktionary/en", "license": "cc:by-sa/4.0", "sources": [{"contributor": "/s/resource/wiktionary/en", "process": "/s/process/wikiparsec/1"}], "weight": 1.0}

    Args:
        file_path: Path to the ConceptNet CSV file (can be gzipped).
        sample_rate: Float between 0.0 and 1.0. The fraction of edges to randomly sample.
                     Defaults to 1.0 (no sampling).

    Returns:
        An rdflib Graph containing the sampled triples (start_node, relation, end_node).
    """
    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError("sample_rate must be between 0.0 and 1.0")

    g = Graph()
    nodes_added = 0
    lines_processed = 0
    # ConceptNet URIs seem relative, let's define a base for rdflib if needed,
    # but URIRef might handle relative paths depending on context.
    # Using them directly for now.
    # base_uri = "http://conceptnet.io"

    try:
        # Handle potential gzipped files transparently
        open_func = open
        if file_path.endswith(".gz"):
            import gzip

            open_func = gzip.open

        with open_func(file_path, "rt", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            print(f"Starting load with sample rate: {sample_rate}...")
            for i, row in enumerate(reader):
                lines_processed = i + 1
                if lines_processed % 1000000 == 0:  # Print progress
                    print(
                        f"Processed {lines_processed // 1000000}M lines, added {nodes_added} triples so far..."
                    )

                # --- Sampling Check ---
                if random.random() >= sample_rate:
                    continue  # Skip this row based on sample rate

                if len(row) == 5:
                    # Fields: edge_uri, relation, start_node, end_node, metadata_json
                    _, rel, start, end, _ = row
                    try:
                        # Convert ConceptNet URIs to rdflib URIRefs
                        # Assuming the URIs provided are sufficient
                        subject = URIRef(start)
                        predicate = URIRef(rel)
                        obj = URIRef(end)
                        g.add((subject, predicate, obj))
                        nodes_added += 1
                    except Exception as e:
                        if (
                            sample_rate == 1.0 or random.random() < 0.01
                        ):  # Only print errors occasionally if sampling heavily
                            print(
                                f"Skipping row {i+1} due to parsing error: {e} | Row: {row}"
                            )
                else:
                    if (
                        sample_rate == 1.0 or random.random() < 0.01
                    ):  # Only print errors occasionally if sampling heavily
                        print(
                            f"Skipping malformed row {i+1}: Expected 5 fields, got {len(row)} | Row: {row}"
                        )

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print(
        f"Finished loading. Processed {lines_processed} lines, added {nodes_added} triples (effective sample rate: {nodes_added/lines_processed if lines_processed > 0 else 0:.4f})."
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
# Let's try sampling 5% of the edges
sampling_fraction = 0.05
print(f"Loading graph with {sampling_fraction*100:.1f}% sampling...")
g = load_conceptnet_csv(
    "data/conceptnet-assertions-5.5.0.csv.gz", sample_rate=sampling_fraction
)
print(f"Total triples in sampled graph: {len(g)}")
# %%
