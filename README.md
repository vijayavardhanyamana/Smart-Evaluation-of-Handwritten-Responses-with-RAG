# Smart-Evaluation-of-Handwritten-Responses-with-RAG

## Introduction
This project automates the evaluation of handwritten responses by integrating HTR (Handwritten Text Recognition) at page level and Retrieval-Augmented Generation (RAG) technologies. It includes advanced text recognition and semantic evaluation for educational assessments.

---

## Steps to Run the Program

1. **Upload Handwritten Answers**:
   - Place the image of the handwritten answer in the `ans_image` folder.
   - Update the image path in the `main.py` file.

2. **Upload Reference Material**:
   - Add the reference PDF to the `Knowledge_Retriever_pdf` folder.
   - Edit the PDF path in the `main.py` file.

3. **Download Required Models**:
   - Download the `models` folder from [this link](<https://drive.google.com/drive/folders/1I1Sb6CxMNrrsu5mM8aiuqfJ-UBl11E0W?usp=sharing>).
   - Place the `models` folder in the same directory as `main.py`.

4. **Update Query**:
   - Modify the query (question) in the `main.py` file to match the evaluation context.

5. **Run the Program**:
   - Execute `main.py` to process the inputs and evaluate the handwritten answers.

---

## Key Features

- Handwritten Text Recognition at the page level and strike word recognition at the word level.
- Knowledge Retrieval powered by RAG with FLAN-T5 and the usage of vector database ChromaDB.
- Model comparison (word embedding models and Sentence Transformer models) for the best semantic analysis.
- Evaluation by multiple scoring mechanisms including:
  - Word embedding models and Sentence Transformers.
  - Metrics like cosine similarity, Word Mover’s Distance (WMD), and soft cosine similarity.
  - TS-GR Score:
    - **Term Significance (TS)**: TS(W) = Number of times word W appears in Answer A / Total number of words in document D.
    - **Global Relevance (GR)**: GR(W) = Number of answers containing word W / Total number of answers.

---
---

## Notes

- Ensure all paths are correctly updated before running the program.
- The `models` folder is essential for text recognition and semantic evaluation.

---
