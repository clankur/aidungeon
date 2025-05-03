# %%
from typing import List, Dict, Any
from world import World
from transformers import pipeline
import torch
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
    def __init__(self) -> None:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        # other model choices
        # FacebookAI/xlm-roberta-large
        # OpenMatch/cocodr-base-msmarco => good retrievers
        # answerdotai/ModernBERT-base
        self.encoder = SentenceTransformer(model_name)
        ner_model_name = "Babelscape/wikineural-multilingual-ner"
        self.ner_model = pipeline(
            "ner", model=ner_model_name, aggregation_strategy="simple"
        )

    def get_relevant_entities(
        self, query: str, world: World, create_entities: bool = False
    ) -> List[str]:
        ner_results = self.ner_model(query)

        # TODO: observe if we need to filter out based on score
        # cursory glance, 0.5 and higher seems reasonable
        entities = set([r["word"] for r in ner_results])

        if create_entities:
            for e in entities:
                if not world.contains(e):
                    world.add_node(e)
        print(f"{entities=}")
        return [e for e in entities if world.contains(e)]

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

    def retrieve(
        self, query: str, world: World, threshold: float = 0.75
    ) -> Dict[str, Dict[str, Any]]:
        query_embedding = self.encode_text(query)
        relevant_entities = self.get_relevant_entities(query, world)
        if not relevant_entities:
            raise ValueError("No relevant entities found in the world")
        # convert world to n_edges x C tensor
        triples = {}

        # TODO: can we query in parallel?
        for entity in relevant_entities:
            for uid, results in world.query(entity).items():
                triples[uid] = {
                    "embedding": [],
                    "relationships": [],
                }
                for _, predicate, object in results:
                    triple = f"{entity} {predicate} {object}"
                    triples[uid]["relationships"].append(triple)
                    triple_embedding = self.encode_text(triple)
                    triple_embedding = einops.rearrange(
                        triple_embedding, "b d -> (b d)"
                    )
                    triples[uid]["embedding"].append(triple_embedding)

                relationship_embeddings = torch.stack(triples[uid]["embedding"])
                logits = einops.einsum(
                    query_embedding,
                    relationship_embeddings,
                    "Q_len C, n_edges C -> Q_len n_edges",
                )
                scores = quiet_softmax(logits)
                scores = einops.rearrange(scores, "1 n_edges -> n_edges")
                triples[uid]["scores"] = scores

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
                    (triples[uid]["relationships"][idx.item()])
                    for idx in selected_indices
                ]
                triples[uid]["retrieved_edges"] = (
                    retrieved_edges,
                    triples[uid]["scores"].sum().item(),
                )

        return triples


#
# use prompt to figure out what is the best statement to add
#   4 potential statements you can add, you argmax to select the best statement
#   with new queries get probabilities
# what pairs of things need new entries, and add their relationship, add it
