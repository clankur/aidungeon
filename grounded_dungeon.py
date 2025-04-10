# %%
from graph import KnowledgeGraph
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel
import einops


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
        model_name = "bert-base-uncased"

        self.graph = graph
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # convert graph to n_edges x C tensor
        triples = []
        self.relationships = []
        for entity in self.graph.graph:
            for relationship in self.graph.graph[entity]:
                triple = (
                    f"{entity} {relationship} {self.graph.graph[entity][relationship]}"
                )
                self.relationships.append(triple)
                triple_embedding = self.encode_text(triple)
                triple_embedding = einops.rearrange(triple_embedding, "b d -> (b d)")
                triples.append(triple_embedding)

        self.relationship_embeddings = torch.stack(triples)

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text into an embedding vector using the sentence transformer model.

        Args:
            text: The text to encode

        Returns:
            torch.Tensor: The embedding vector
        """

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        embedding = einops.rearrange(embedding, "b d -> b d")
        return embedding

    def retrieve(self, query: str, threshold: float = 0.75) -> List[str]:
        retrieved_edges = []
        query_embedding = self.encode_text(query)
        print(query_embedding.shape)
        print(self.relationship_embeddings.shape)

        logits = einops.einsum(
            query_embedding,
            self.relationship_embeddings,
            "Q_len D, n_edges D -> Q_len n_edges",
        )

        scores = quiet_softmax(logits)
        print(scores)
        print(self.relationships)

        return retrieved_edges


# %%
edges = [
    ("Bob", "owns", "Jason"),
    ("Bob", "owns", "Carl"),
    ("Bob", "married", "Alice"),
    ("Bob", "on_torso", "Shirt that says 'I heart cats'"),
    ("Bob", "on_torso", "Pants"),
    ("Jessica", "owns", "Doug"),
    ("Jessica", "owns", "Kate"),
    ("Jessica", "on_head", "Hat"),
    ("Jessica", "on_torso", "Shirt"),
    ("Jessica", "on_waist", "Pants"),
]

kg = KnowledgeGraph.build_graph(edges)
print(kg.graph)


retriever = Retriever(kg)
retriever.retrieve("Who is Bob's wife?")

# %%
retriever.retrieve("Jessica loves dogs")

# %%
# alternate retrieval approaches
#   cross attend between edges and the query and select the edges with High Scores (?)
#   leverage graph spatial structure (?), use GNN (???)

# We might want to not use all edges for logits
# instead we retrieve edges with entities relevant (?) to query
# use like Entity Extraction to get them and then filter upon those edges?
# %%
