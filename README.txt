
# README: Running and Evaluating Models for Q&A
Sai Charan Somineni

## Overview
This project involves running three models sequentially (BERT, GPT-2, and Groq) to generate answers to 15 different questions, consisting of "what," "who," "how," "when," and "why" types. Human evaluation determines the best-performing model.

---

## Prerequisites
1. Python 3.8 or later installed.
2. Required libraries: `transformers`, `sentence-transformers`, `faiss`, `numpy`, `torch`, `logging`, `json`, and `requests`.
3. GPU recommended for faster processing.

Install dependencies:
```bash
pip install transformers sentence-transformers faiss-cpu numpy torch requests pdfplumber
```

---

## Steps to Run

### Step 1: Extraction
Run `Extraction_final.py` to download and process PDFs:
```bash
python Extraction_final.py
```
- **Input**: URLs of PDFs defined in the script.
- **Output**: `pavement_data.json` containing extracted text and images.
- **Algorithm**: The script uses PyMuPDF (fitz) for text extraction, and the PIL library is used for image processing. Relevance is determined based on pre-configured keywords and image quality thresholds.

---

### Step 2: Running Models
#### BERT
```bash
python Bert_final.py
```
- **Description**: Uses BERT to answer questions by summarizing retrieved chunks and providing comprehensive answers with relevant images.
- **Algorithm**:
  - FAISS is used to retrieve relevant content based on a query.
  - Sentence embeddings are generated using the `SentenceTransformer` model.
  - BERT (fine-tuned on SQuAD) is used for question-answering tasks, and summarization is performed using `facebook/bart-large-cnn`.
- **Output**: Logs answers and relevant images in `chatbot_log_BERT.txt`.

#### GPT-2
```bash
python GPT2.py
```
- **Description**: Employs GPT-2 to generate answers based on the context retrieved from the `pavement_data.json`.
- **Algorithm**:
  - FAISS retrieves relevant content using `SentenceTransformer` embeddings.
  - GPT-2 generates answers by utilizing retrieved context and an input prompt.
- **Output**: Logs answers and related images in `chatbot_GPT2_log.txt`.

#### Groq
```bash
python Groq.py
```
- **Description**: Integrates Groq's advanced API to generate answers and retrieve related images.
- **Algorithm**:
  - FAISS retrieves content using `SentenceTransformer` embeddings.
  - Groq’s `llama3-8b-8192` model generates responses based on the prompt containing context and questions.
- **Output**: Logs answers and related images in `groq_chatbot_log.txt`.

---

## Interaction
For all models, run the scripts interactively to input questions:
- Example:
  ```
  You: What is Ground Penetrating Radar?
  Chatbot: Ground Penetrating Radar is a non-destructive testing technique used...
  ```

---

### Step 3: Generate Interactions for Evaluation
1. Prepare 15 questions covering "what," "who," "how," "when," and "why."
2. Interact with each model to generate responses for the questions.
3. Save the outputs for comparison.

---

### Step 4: Evaluate Outputs
- Conduct human evaluation to assess the quality of responses based on:
  - Relevance
  - Completeness
  - Language clarity
- Choose the best-performing model. In this case, **BERT** was determined to be the best model.

---

### Notes
- Ensure the `pavement_data.json` is present in the directory before running any model scripts.
- Modify the logging configurations in each script for detailed output tracking if required.

--- 
## References

1. Hugging Face - BERT Documentation: https://huggingface.co/transformers/v3.0.2/model_doc/bert.html
2. Hugging Face - GPT-2 Documentation:https://huggingface.co/transformers/v3.0.2/model_doc/gpt2.html
3. FAISS Documentation:https://faiss.ai/index.html
4. LangChain Documentation: https://python.langchain.com/docs/introduction/
5. Groq Platform Documentation: https://console.groq.com/docs/overview

--- 
