# %%
from graph import KnowledgeGraph
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import einops
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F


# %%
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
        model_name = "OpenMatch/cocodr-base-msmarco"
        # other model choices
        # FacebookAI/xlm-roberta-large
        # OpenMatch/cocodr-base-msmarco => good retrievers
        # answerdotai/ModernBERT-base
        self.graph = graph
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        input = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = self.encoder(**input)
            embedding = output.last_hidden_state[:, 0]  # CLS token
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def retrieve(self, query: str, threshold: float = 0.75) -> List[str]:
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

        sorted_indices = torch.argsort(scores, descending=True)
        sorted_scores = scores[sorted_indices]
        cumulative_scores = torch.cumsum(sorted_scores, dim=0)
        mask = cumulative_scores <= threshold

        # Add one more element after threshold is reached
        if torch.any(~mask):
            first_false_idx = torch.where(~mask)[0][0]
            if first_false_idx > 0:  # Ensure we don't go out of bounds
                mask[first_false_idx] = True

        selected_indices = sorted_indices[mask]

        retrieved_edges = [
            (relationships[idx.item()], scores[idx].item()) for idx in selected_indices
        ]

        return retrieved_edges


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
    ("Bob", "enemy", "Steve"),
    ("Jessica", "pet", "Doug"),
    ("Jessica", "pet", "Kate"),
    ("Jessica", "spouse", "Bob"),
    ("Jessica", "head", "Hat"),
    ("Jessica", "torso", "Shirt"),
    ("Jessica", "waist", "Pants"),
    ("Jessica", "enemy", "Steve"),
    ("Steve", "enemy", "Bob"),
    ("Steve", "enemy", "Jessica"),
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

# %%
retriever.retrieve("Who is Bob's enemy?")

# %%
retriever.retrieve("Who is Steve's enemy?")

# alternate retrieval approaches
#   cross attend between edges and the query and select the edges with High Scores (?)
#   leverage graph spatial structure (?), use GNN (???)
# %%
