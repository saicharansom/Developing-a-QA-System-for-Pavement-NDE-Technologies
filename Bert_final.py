import json
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import warnings
import logging

# Configure logging
logging.basicConfig(
    filename='chatbot_log_BERT.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Step 1: Load and Chunk the JSON Data
def load_and_chunk_json(file_path, chunk_size=300):
    with open(file_path, 'r') as f:
        data = json.load(f)
    chunks = []
    chunk_metadata = []  # To store metadata like images
    for item in data:
        title = item.get('title', '')
        content = item.get('content', '')
        images = item.get('images', [])
        chunked_content = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        for chunk in chunked_content:
            chunks.append((title, chunk))
            chunk_metadata.append(images)  # Associate images with chunks
    return chunks, chunk_metadata

# Step 2: Create and Populate FAISS Index
def create_faiss_index(chunks, embedding_model):
    embeddings = [embedding_model.encode(chunk[1]) for chunk in chunks]
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

# Step 3: Query FAISS Index
def query_index(index, embedding_model, query, chunks, chunk_metadata, top_k=3):
    query_vector = embedding_model.encode([query]).astype('float32')
    _, indices = index.search(query_vector, top_k)
    return [(chunks[i], chunk_metadata[i]) for i in indices[0]]

# Step 4: Summarize Retrieved Chunks
def summarize_chunks(chunks, summarizer):
    combined_text = " ".join([chunk[0][1] for chunk in chunks])
    max_len = min(len(combined_text.split()), 600)  # Adjust max_length dynamically
    summary = summarizer(combined_text, max_length=max_len, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Step 5: Generate Comprehensive Answer with Relevant Images
def generate_comprehensive_answer(qa_pipeline, summarizer, retrieved_chunks, query):
    summarized_context = summarize_chunks(retrieved_chunks, summarizer)
    enriched_context = summarized_context + " " + " ".join([chunk[0][1] for chunk in retrieved_chunks])
    qa_model_output = qa_pipeline(question=query, context=enriched_context)
    qa_answer = qa_model_output['answer']
    
    # Fallback for incomplete answers
    if len(qa_answer.split()) < 5:
        qa_answer = summarized_context

    # Filter relevant images: only use images directly linked to the retrieved chunks
    relevant_images = []
    for chunk, images in retrieved_chunks:
        relevant_images.extend(images)

    # Remove duplicates
    relevant_images = list(set(relevant_images))

    # Log the interaction
    logging.info(f"Query: {query}")
    logging.info(f"Retrieved Chunks: {[chunk[0] for chunk, _ in retrieved_chunks]}")
    logging.info(f"Answer: {qa_answer}")
    logging.info(f"Relevant Images: {relevant_images}")

    return qa_answer, relevant_images

# Step 6: Interactive Chat Function
def interactive_chat_with_comprehensive_answers():
    print("Chatbot is ready with enhanced answers and relevant images! Type 'exit' to quit.")
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        retrieved_chunks = query_index(faiss_index, embedding_model, user_query, chunk_data, chunk_metadata)
        answer, images = generate_comprehensive_answer(qa_pipeline, summarizer, retrieved_chunks, user_query)
        print(f"Chatbot: {answer}")
        if images:
            print("\nRelated Images:")
            for img in images:
                print(f"- {img}")

# Main Script
if __name__ == '__main__':
    # Suppress non-critical warnings
    warnings.filterwarnings('ignore')
    
    # Check if CUDA is available
    device =  device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models with CUDA support
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    qa_pipeline = pipeline('question-answering', model="bert-large-uncased-whole-word-masking-finetuned-squad", device=device)
    summarizer = pipeline('summarization', model="facebook/bart-large-cnn", device=device)
    
    # Load data and create FAISS index
    json_file_path = 'pavement_data.json'  # Update with your JSON file path
    chunks, chunk_metadata = load_and_chunk_json(json_file_path)
    faiss_index, chunk_data = create_faiss_index(chunks, embedding_model)

    # Start chatbot with enhanced answer generation
    interactive_chat_with_comprehensive_answers()
