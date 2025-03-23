import json
import logging
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# Configure logging
logging.basicConfig(
    filename='groq_chatbot_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load JSON data
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Initialize models
def initialize_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    client = Groq(api_key="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")  
    return embedding_model, client

# Create FAISS index
def create_faiss_index(data, embedding_model):
    embedding_dim = 384  # Dimension of embeddings from MiniLM
    index = faiss.IndexFlatL2(embedding_dim)
    embeddings = [embedding_model.encode(chunk['content'], convert_to_numpy=True) for chunk in data]
    index.add(np.array(embeddings, dtype='float32'))
    return index

# Retrieve relevant chunks
def retrieve_relevant_chunks(query, embedding_model, index, data, top_k=3):
    query_embedding = embedding_model.encode(query, convert_to_numpy=True).reshape(1, -1)
    _, indices = index.search(query_embedding, top_k)
    return [data[i] for i in indices[0] if i < len(data)][:top_k]

# Trim context to fit within token limits
def trim_context(context, max_tokens):
    tokens = context.split()
    if len(tokens) > max_tokens:
        trimmed_tokens = tokens[-max_tokens:]  # Keep the most recent tokens
        return " ".join(trimmed_tokens)
    return context

# Generate answer and related images
def generate_answer_and_images(query, embedding_model, client, index, data, max_context_tokens=2048, max_new_tokens=300):
    # Retrieve relevant chunks
    retrieved_chunks = retrieve_relevant_chunks(query, embedding_model, index, data)

    # Compile context from retrieved chunks
    context = " ".join([chunk['content'] for chunk in retrieved_chunks])

    # Collect relevant images (ensure uniqueness)
    images = []
    for chunk in retrieved_chunks:
        if 'images' in chunk:
            images.extend(chunk['images'])

    # Deduplicate and limit images
    images = list(set(images))[:5]

    # Trim the context to fit the token limit
    context = trim_context(context, max_context_tokens - max_new_tokens)

    # Prepare the prompt
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    logging.info(f"Prompt Token Count: {len(prompt.split())}")

    # Send the request to the Groq model
    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=max_new_tokens,
            top_p=0.95
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error querying Groq API: {e}")
        answer = "I'm sorry, I encountered an error while processing your query."

    # Log the query, context, answer, and images
    logging.info(f"Query: {query}")
    logging.info(f"Retrieved Chunks: {[chunk['content'][:50] + '...' for chunk in retrieved_chunks]}")
    logging.info(f"Answer: {answer}")
    logging.info(f"Images: {images}")

    return answer, images

# Interactive chatbot
def interactive_chat(data, embedding_model, client, index):
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Generate response and related images
        answer, images = generate_answer_and_images(query, embedding_model, client, index, data)

        # Display answer
        print(f"Chatbot: {answer}")

        # Display related images
        if images:
            print("\nRelated Images:")
            for img in images:
                print(f"- {img}")

# Main script
if __name__ == '__main__':
    # Load data
    json_file_path = 'pavement_data.json'  
    data = load_json(json_file_path)

    # Initialize models and FAISS index
    embedding_model, client = initialize_models()
    index = create_faiss_index(data, embedding_model)

    # Start chatbot
    interactive_chat(data, embedding_model, client, index)
