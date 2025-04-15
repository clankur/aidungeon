# %%
from rdflib import Graph, URIRef
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import einops
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from typing import Tuple


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


def get_en_pred_obj(pred_obj: Tuple[str, str]) -> str:
    """
    converts a tuple of (pred_obj, score) to a string of the form. ie).
    (/r/IsA, /c/en/president_of_united_states) -> "IsA president of united states"
    """
    pred_uri, obj_uri = pred_obj

    pred_name = str(pred_uri).split("/")[-1]
    obj_name = str(obj_uri).replace("/c/en/", "").replace("_", " ")

    return f"{pred_name} {obj_name}"


# %%
class KnowledgeGraph:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def query(self, query: str) -> List[str]:
        query = query.replace(" ", "_").lower()
        query_uri = URIRef(f"/c/en/{query}")
        query_string_props = f"""
            SELECT ?p ?o
            WHERE {{
            <{query_uri}> ?p ?o .
        }}
    """
        return self.graph.query(query_string_props)

    def contains(self, query: str) -> bool:
        query = query.replace(" ", "_").lower()
        query_uri = URIRef(f"/c/en/{query}")
        ask_query = f"ASK {{ <{query_uri}> ?p ?o . }}"
        results = self.graph.query(ask_query)
        return bool(results)


# %%
class Retriever:
    def __init__(self, graph: KnowledgeGraph) -> None:
        model_name = "OpenMatch/cocodr-base-msmarco"
        # other model choices
        # FacebookAI/xlm-roberta-large
        # OpenMatch/cocodr-base-msmarco => good retrievers
        # answerdotai/ModernBERT-base
        self.graph = KnowledgeGraph(graph)

        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        ner_model_name = "dslim/bert-base-NER"
        self.ner_model = pipeline("ner", model=ner_model_name)

    def get_relevant_entities(self, query: str) -> List[str]:
        ner_results = self.ner_model(query)

        entities = []
        current = ""
        for r in ner_results:
            tag = r["entity"]
            word = r["word"]

            if tag.startswith("B-"):
                if current:
                    entities.append(current.strip())
                current = word if not word.startswith("##") else word[2:]
            elif tag.startswith("I-"):
                if word.startswith("##"):
                    current += word[2:]
                else:
                    current += " " + word
            else:
                if current:
                    entities.append(current.strip())
                    current = ""

        if current:
            entities.append(current.strip())

        print(f"{entities=}")
        return [e for e in entities if self.graph.contains(e)]

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

        # TODO: adjust so encodings done in parallel
        for entity in relevant_entities:
            for predicate, object in self.graph.query(entity):
                pred_obj = get_en_pred_obj((predicate, object))
                triple = f"{entity} {pred_obj}"
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
