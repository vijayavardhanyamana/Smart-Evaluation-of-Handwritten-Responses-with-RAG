import fitz  # PyMuPDF
import re
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import os



def clean_text(text):
    # Keep only letters, numbers, punctuation, whitespace, and newlines
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"()\-]", "", text)
    return cleaned_text

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_text = page.get_text()
            cleaned_text = clean_text(page_text)
            text += cleaned_text
    return text

def clean_data(text):
    cleaned_text = re.sub(r'\n{2,}', '. \n', text)  # Replace multiple newlines with a single newline
    cleaned_text = re.sub(r' {2,}', '. \n', cleaned_text)  # Replace multiple spaces with a newline
    
    return cleaned_text.strip()  # Strip leading/trailing whitespace


def combine_list(strings):
    combined_list = [] 
    current_combined = ""
    for string in strings:
        word_count = len(string.split())
        
        if len(current_combined.split()) < 20:
            current_combined += " " + string.strip()  # Adding space before new string
            
        # If the combined string reaches at least 20 words, add it to the final list
        if len(current_combined.split()) >= 20:
            combined_list.append(current_combined)  # Strip to remove leading/trailing whitespace
            current_combined = ""  # Reset for the next round
    if current_combined:
        combined_list.append(current_combined.strip())
    return combined_list


def create_databse(data,name):
    # Initialize the Persistent Client
    client = chromadb.PersistentClient(path="correct_answer_generation/chroma_db")
    
    collections = client.list_collections()
    if name in collections:
        print("change the pdf name")
        return 

    # Create a Collection
    collection = client.create_collection(name)

    # List of strings
    strings = data

    # Generate embeddings using SentenceTransformer
    similarity_model_path = "models/similarity_model"
#     tokenizer = AutoTokenizer.from_pretrained(similarity_model_path)
    model = SentenceTransformer(similarity_model_path)

    embeddings = model.encode(strings)  # Generate embeddings for the list of strings

    # Create documents and add them to the collection
    unique_id =[]
    for i in range(len(embeddings)):
        unique_id.append(str(uuid.uuid4()))
    
    collection.add(
    documents=strings,
    ids=unique_id
    )


    print("Documents added to the collection.")

    

def create_database_main(path):
    
    pdf_path = path
    pdf_text = extract_text_from_pdf(pdf_path)
    data = clean_data(pdf_text)
    data = data.split('. \n')
    for i in range(len(data)):
        data[i] = re.sub(r' \n', ' ', data[i])
        data[i] = re.sub(r'\s+', ' ', data[i])
    data = [text for text in data if len(text) >= 2]
    data = combine_list(data)
    
    name = os.path.splitext(os.path.basename(path))[0]

    path = path.replace("/", "_")
    create_databse(data, path)
    
