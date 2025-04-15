# %%
from rdflib import Graph, term, RDFS, Literal
from rdflib.exceptions import ParserError
import time
from typing import Union, List


# %%
def load_graph(file_path: str) -> Graph:
    # Create an empty RDF graph
    g = Graph()

    try:
        # Attempt to parse the Turtle file
        # The format='turtle' argument tells rdflib which syntax to expect
        start_time = time.time()
        g.parse(file_path, format="turtle")
        end_time = time.time()
        print(f"Successfully parsed {len(g)} triples from {file_path}")
        print(f"Time taken to parse: {end_time - start_time} seconds")
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
    except ParserError as e:
        print(f"Error parsing TTL file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return g


# %%
file_path = "data/wikidata_cc_full_3_hop.ttl"
g = load_graph(file_path)


# %%
# need to lookup a URIRef from a label
# then lookup the URIRef's edges
# maybe lookup the labels associated to the edges
def get_uris_for_label(label_str: str) -> List[term.URIRef]:
    """
    Looks up a string label in the graph and returns a list of URIRefs
    associated with that label (typically via rdfs:label).
    """
    # Create a Literal object for the label string.
    # Consider adding language tags if necessary, e.g., Literal(label_str, lang='en')
    label_literal = Literal(label_str)

    # SPARQL query to find subjects (?s) that have the given label
    # Using rdfs:label is common, but other properties might be used as well.
    # We also need to handle cases where the label might have a language tag.
    # This query looks for an exact match of the literal.
    query_string = f"""
        SELECT DISTINCT ?s ?p
        WHERE {{
            ?s ?p {label_literal.n3()} .
        }}
    """

    uris = []
    try:
        results = g.query(query_string)
        for row in results:
            # Ensure the result is a URIRef before adding
            if isinstance(row.s, term.URIRef):
                uris.append((row.s, row.p))
        print(f"Found {len(uris)} URIs for label '{label_str}'")
    except Exception as e:
        print(f"An error occurred during query for label '{label_str}': {e}")

    return uris


def query_uri(query_uri: term.URIRef) -> List[term.URIRef]:
    print(f"\nQuerying for nodes connected TO {query_uri}...")
    query_string_obama_props = f"""
        SELECT ?p ?o
        WHERE {{
            <{query_uri}> ?p ?o .
        }}
    """
    results_obama_props = g.query(query_string_obama_props)

    return results_obama_props


# %%
label_to_find = "George Washington"
print(f"\nLooking up URIs for label: '{label_to_find}'...")
found_uris = get_uris_for_label(label_to_find)
for uri_tuple in found_uris:
    subject_uri, predicate_uri = uri_tuple
    print(f"\t{subject_uri} -> {predicate_uri} -> {label_to_find}")
    query_results = query_uri(subject_uri)
    print(
        f"\tFound {len(query_results)} outgoing properties/objects for {subject_uri}:"
    )
    for row in query_results:
        predicate_uri = row.p
        object_val = row.o

        # Look up the label for the predicate URI
        # g.label() searches for skos:prefLabel and rdfs:label
        predicate_label = g.label(predicate_uri)

        # Use the label if found, otherwise default to the URI string
        predicate_display = predicate_label if predicate_label else str(predicate_uri)

        # Also try to get label for the object if it's a URI
        object_display = object_val
        if isinstance(object_val, term.URIRef):
            object_label = g.label(object_val)
            if object_label:
                object_display = object_label  # Optionally add URI in parens: f"{object_label} ({object_val})"

        print(f"\t\t{predicate_display} -> {object_display}")

# %%
