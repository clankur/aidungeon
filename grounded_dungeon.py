# %%
from graph import KnowledgeGraph
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import einops
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F


# %%
def cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """
    Calculate the cosine similarity between two vectors.

    Args:
        vec_a: First vector
        vec_b: Second vector

    Returns:
        float: Cosine similarity between the two vectors
    """
    dot_product = torch.dot(vec_a, vec_b)
    magnitude_a = torch.norm(vec_a)
    magnitude_b = torch.norm(vec_b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def quiet_softmax(logits: torch.Tensor) -> torch.Tensor:
    """softmax with 0 logit padding, drop logits that have negative alignment"""
    # Create a tensor of zeros with the same shape as the max_logits
    zero_tensor = torch.zeros_like(einops.reduce(logits, "Qlen Klen -> Qlen 1", "max"))
    max_logits = torch.maximum(
        einops.reduce(logits, "Qlen Klen -> Qlen 1", "max"), zero_tensor
    )
    stable_logits = logits - max_logits
    probs = (torch.exp(stable_logits)) / (
        torch.exp(-max_logits)
        + einops.reduce(
            torch.exp(stable_logits),
            "Qlen Klen -> Qlen 1",  # Fixed dimension reduction
            "sum",
        )
    )
    return probs


# %%
class Retriever:
    def __init__(self, graph: KnowledgeGraph) -> None:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        # other model choices
        # FacebookAI/xlm-roberta-large
        # OpenMatch/cocodr-base-msmarco => good retrievers
        # answerdotai/ModernBERT-base
        self.graph = graph
        self.encoder = SentenceTransformer(model_name)

        ner_model_name = "dslim/bert-base-NER"
        self.ner_model = pipeline("ner", model=ner_model_name)

    def get_relevant_entities(self, query: str) -> List[str]:
        ner_results = self.ner_model(query)
        return [
            result["word"]
            for result in ner_results
            if result["word"] in self.graph.graph
        ]

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text into an embedding vector using the sentence-transformer model.

        Args:
            text: The text to encode

        Returns:
            torch.Tensor: The embedding vector
        """
        embedding = self.encoder.encode(text, convert_to_tensor=True)
        embedding = einops.rearrange(embedding, "C -> 1 C")
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def retrieve(self, query: str, threshold: float = 0.75) -> List[str]:
        retrieved_edges = []
        query_embedding = self.encode_text(query)
        relevant_entities = self.get_relevant_entities(query)
        if not relevant_entities:
            raise ValueError("No relevant entities found in the graph")
        # convert graph to n_edges x C tensor
        triples = []
        relationships = []
        for entity in relevant_entities:
            for relationship in self.graph.graph[entity]:
                triple = (
                    f"{entity} {relationship} {self.graph.graph[entity][relationship]}"
                )
                relationships.append(triple)
                triple_embedding = self.encode_text(triple)
                triple_embedding = einops.rearrange(triple_embedding, "b d -> (b d)")
                triples.append(triple_embedding)

        relationship_embeddings = torch.stack(triples)

        logits = einops.einsum(
            query_embedding,
            relationship_embeddings,
            "Q_len C, n_edges C -> Q_len n_edges",
        )

        scores = quiet_softmax(logits)
        scores = einops.rearrange(scores, "1 n_edges -> n_edges")
        scores = scores.tolist()

        # sort relationships by scores
        sorted_relationships = [
            (score, relationship)
            for score, relationship in sorted(
                zip(scores, relationships), key=lambda x: -x[0]
            )
        ]
        return sorted_relationships


#
# use prompt to figure out what is the best statement to add
#   4 potential statements you can add, you argmax to select the best statement
#   with new queries get probabilities
# what pairs of things need new entries, and add their relationship, add it


# %%
edges = [
    ("Bob", "pet", "Jason"),
    ("Bob", "pet", "Carl"),
    ("Bob", "spouse", "Jessica"),
    ("Bob", "torso", "Shirt'"),
    ("Bob", "waist", "Pants"),
    ("Jessica", "pet", "Doug"),
    ("Jessica", "pet", "Kate"),
    ("Jessica", "spouse", "Bob"),
    ("Jessica", "head", "Hat"),
    ("Jessica", "torso", "Shirt"),
    ("Jessica", "waist", "Pants"),
]

kg = KnowledgeGraph.build_graph(edges)
print(kg.graph)


retriever = Retriever(kg)
retriever.retrieve("Who is Bob's dog?")

# %%
retriever.retrieve("Jessica loves dogs")

# %%
retriever.retrieve("What is Bob wearing?")
# %%
retriever.retrieve("Who is Bob's spouse?")

# %%
retriever.retrieve("Jessica is Bob's what?")

# %%
retriever.retrieve("What is Jessica wearing?")


# alternate retrieval approaches
#   cross attend between edges and the query and select the edges with High Scores (?)
#   leverage graph spatial structure (?), use GNN (???)

# We might want to not use all edges for logits
# instead we retrieve edges with entities relevant (?) to query
# use like Entity Extraction to get them and then filter upon those edges?
# %%
