import os
import gc
import torch
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate


def setup_rag_system():
    # Print memory warning
    print("Warning: Loading large language models on CPU can be memory-intensive.")
    print("Make sure you have enough RAM available.")

    # Load documents from a directory
    documents = SimpleDirectoryReader("./data").load_data()

    # Create a node parser for chunking documents
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)

    # Create a Hugging Face LLM with CPU optimization settings
    llm = HuggingFaceLLM(
        model_name="EVA-UNIT-01/EVA-Qwen2.5-1.5B-v0.0",
        tokenizer_name="EVA-UNIT-01/EVA-Qwen2.5-1.5B-v0.0",
        context_window=2048,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.7, "do_sample": True},
        device_map="auto",  # Let HF decide best device mapping
        model_kwargs={
            "torch_dtype": torch.float32,  # Use fp32 instead of fp16
            "low_cpu_mem_usage": True,  # Enable low CPU memory usage
        },
    )

    # Create a Hugging Face embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # Explicitly use CPU
        cache_folder="./model_cache",  # Store models locally to avoid repeated downloads
    )

    # Configure global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser

    # Create a vector store index from the documents
    index = VectorStoreIndex.from_documents(documents)

    # Create a retriever with similarity threshold
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=2,  # Retrieve top 2 most similar chunks
        similarity_threshold=0.7,  # Only retrieve if similarity is above this threshold
    )

    # Create a proper PromptTemplate object for the QA template
    prompt = """
    You're a dungeon master and storyteller that provides any kind of game, roleplaying and story content.\n\n
    
    Instructions:\n\n
    
    - Be specific, literal, concrete, creative, grounded and clear\n
    - Continue the text where it ends without repeating\n
    - Avoid reusing themes, sentences, dialog or descriptions\n
    - Continue unfinished sentences\n- > means an action attempt; it is forbidden to output >\n
    - Show realistic consequences
    """
    qa_template = PromptTemplate(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query. If the context doesn't contain relevant information "
        "to answer the query, respond with 'I don't have enough information to answer this question.'\n"
        "Query: {query_str}\n"
        "Answer: "
    )

    # Create a response synthesizer with a "no answer" template
    response_synthesizer = get_response_synthesizer(
        response_mode="compact",
        text_qa_template=qa_template,
    )

    # Create a custom query engine directly from the retriever
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    return query_engine


def query_rag(query_engine, question):
    # Query the system
    response = query_engine.query(question)
    return response


def main():
    # Setup the RAG system
    print("Setting up RAG system...")
    query_engine = setup_rag_system()

    # Example usage
    while True:
        # Force garbage collection to free up memory
        gc.collect()

        question = input("\nEnter your question (or 'exit' to quit): ")
        if question.lower() == "exit":
            break

        print("\nQuerying the RAG system...")
        try:
            response = query_rag(query_engine, question)
            print(f"\nResponse: {response}")
        except Exception as e:
            print(f"Error during query: {str(e)}")
            print(
                "This may be due to memory constraints. Try restarting the program or simplifying your query."
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        print("If you're experiencing memory issues, consider:")
        print("1. Reducing the model size")
        print("2. Using a machine with more RAM")
        print("3. Breaking your data into smaller chunks")
    finally:
        # Clean up torch cache
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
