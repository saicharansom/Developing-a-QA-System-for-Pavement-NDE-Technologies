from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import torch
import logging

# Configure logging
logging.basicConfig(
    filename='chatbot_GPT2_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load JSON data
file_path = 'pavement_data.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Initialize models and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_model.config.pad_token_id = gpt2_tokenizer.eos_token_id

# Load FAISS index with embeddings
embedding_dim = 384  # Dimension of embeddings from MiniLM
index = faiss.IndexFlatL2(embedding_dim)
embeddings = [embedding_model.encode(chunk['content'], convert_to_numpy=True) for chunk in data]
index.add(np.array(embeddings, dtype='float32'))

def retrieve_relevant_chunks(query, top_k=3):
    """
    Retrieve relevant chunks from the dataset based on the query.
    """
    query_embedding = embedding_model.encode(query, convert_to_numpy=True).reshape(1, -1)
    _, indices = index.search(query_embedding, top_k)
    return [data[i] for i in indices[0] if i < len(data)]

def trim_context_to_fit(context, max_tokens=900):
    """
    Trim the context to fit within the maximum token limit for GPT-2.
    """
    tokens = gpt2_tokenizer.encode(context)
    if len(tokens) > max_tokens:
        trimmed_tokens = tokens[:max_tokens]
        return gpt2_tokenizer.decode(trimmed_tokens, skip_special_tokens=True)
    return context

def generate_answer_and_images(query, max_tokens=1024, max_new_tokens=200):
    """
    Generate an answer to the query using GPT-2 and retrieve relevant images.
    """
    # Retrieve relevant chunks
    retrieved_chunks = retrieve_relevant_chunks(query)
    
    # Compile context from retrieved chunks
    context = " ".join([chunk['content'] for chunk in retrieved_chunks])
    
    # Collect relevant images (ensure uniqueness)
    images = []
    for chunk in retrieved_chunks:
        if 'images' in chunk:
            images.extend(chunk['images'])

    # Deduplicate images and limit the number of images
    images = list(set(images))[:5]  # Limit to top 5 images

    # Prepare GPT-2 prompt
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    tokens = gpt2_tokenizer.encode(prompt, return_tensors="pt").to(device)
    if tokens.size(1) > max_tokens - max_new_tokens:
        tokens = tokens[:, -(max_tokens - max_new_tokens):]

    # Generate GPT-2 output
    output = gpt2_model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        pad_token_id=gpt2_tokenizer.eos_token_id
    )

    # Decode and clean the output
    output_text = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    if "Answer:" in output_text:
        answer = output_text.split("Answer:")[-1].strip()
    else:
        answer = output_text.strip()

    # Log the query and response details
    logging.info(f"Query: {query}")
    logging.info(f"Retrieved Chunks: {[chunk['content'][:50] + '...' for chunk in retrieved_chunks]}")
    logging.info(f"Answer: {answer}")
    logging.info(f"Images: {images}")

    return answer, images


def interactive_chat():
    """
    Interactive chatbot with improved image handling.
    """
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        
        answer, images = generate_answer_and_images(query)
        print(f"Chatbot: {answer}")
        
        if images:
            print("\nRelated Images:")
            for img in images:
                print(f"- {img}")


# Start chatbot
interactive_chat()
