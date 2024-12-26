from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
import faiss
import pdfplumber
import requests
import os
import pickle

app = Flask(__name__)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size):
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Function to generate embeddings
def generate_embeddings(chunks, embedding_model):
    embeddings = embedding_model.encode(chunks, convert_to_tensor=True, device="cuda")
    return embeddings

# Function to create FAISS index
def create_faiss_index(embeddings):
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings.cpu().numpy())  # Move embeddings to CPU before adding
    return index

# Function to save FAISS index and chunks
def save_faiss_index(index, chunks, index_path, chunks_path):
    faiss.write_index(index, index_path)
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)

# Function to load FAISS index and chunks
def load_faiss_index(index_path, chunks_path):
    index = faiss.read_index(index_path)
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    return index, chunks

# Function to search FAISS index
def search_faiss_index(query, index, chunks, embedding_model, k=3):
    query_embedding = embedding_model.encode([query], convert_to_tensor=True, device="cuda").detach().cpu().numpy()
    distances, indices = index.search(query_embedding, k)
    return [(chunks[i], distances[0][j]) for j, i in enumerate(indices[0])]

# API request function
def send_request_to_mrc(query, passages, mrc_url):
    payload = {
        "query": {
            "paraphrased": [query]
        },
        "passages": [
            {
                "record_id": f"id-{i}",
                "title": "",
                "description": passage,
                "search_query": query
            }
            for i, passage in enumerate(passages)
        ],
        "lang": "ko", # select input language ko/en
        "threshold": 0,
        "abstract": True,
        "abstract_length": 1000,
        "search_fields": ["title", "content"]
    }
    response = requests.post(mrc_url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

# Load embedding model
EMBEDDING_MODEL_NAME = "" # Embedding model from huggingface
mrc_url = '' # MRC API link

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda")

# Paths for FAISS index and chunks
index_path = "faiss_index.bin"
chunks_path = "chunks.pkl"

# Control whether to vectorize
VECTORIZE = True  # 벡터화를 수행할지 여부 (True/False)

if VECTORIZE:
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        # Load existing data
        index, chunks = load_faiss_index(index_path, chunks_path)
        print("Load existing faiss index")
    else:
        # Create new index and chunks
        index = None
        chunks = []
        print("No existing faiss index, instead saving new faiss index")

    # Process new PDF
    new_pdf_path = ""  # 벡터화 할 PDF 경로
    new_text = extract_text_from_pdf(new_pdf_path)
    new_chunks = split_text_into_chunks(new_text, chunk_size=1000)
    new_embeddings = generate_embeddings(new_chunks, embedding_model)

    if index is None:
        # Create new index
        index = create_faiss_index(new_embeddings)
        chunks = new_chunks
    else:
        # Merge with existing data
        index.add(new_embeddings.cpu().numpy())
        chunks.extend(new_chunks)

    # Save updated data
    save_faiss_index(index, chunks, index_path, chunks_path)
    print("Appending new pdf's faiss index completed.")
else:
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        # Load existing data only
        index, chunks = load_faiss_index(index_path, chunks_path)
        print("Just Load existing faiss index, No vectorization")
    else:
        print("No existing faiss index. Set 'VECTORIZE = True' to create new faiss index.")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query = request.form['query']

        # Search FAISS index
        results = search_faiss_index(query, index, chunks, embedding_model, k=3)
        top_passages = [passage for passage, _ in results]

        # Send request to MRC API
        try:
            mrc_response = send_request_to_mrc(query, top_passages, mrc_url)
            answers = [
                {
                    "passage": data.get("description", "No description found"),
                    "similarity": results[i][1],
                    "answer": data.get("answer", "No answer found")
                }
                for i, data in enumerate(mrc_response.get("data", []))
            ]
            return render_template('results.html', query=query, answers=answers)
        except Exception as e:
            return render_template('error.html', error=str(e))

    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9900, debug=True)
