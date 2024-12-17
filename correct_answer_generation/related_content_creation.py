import chromadb


def doc_creation(q,collection_name):

    client_ = chromadb.PersistentClient(path="correct_answer_generation/chroma_db")

    # collection = client_.get_collection(name='OperatingSystems')
    collection = client_.get_collection(name=collection_name)


    results = collection.query(
    query_texts=[q],
    n_results=7 # how many results to return
    )

    data = ""
    for i in results['documents'][0]:
        data += " "+i
        print(i)
    return data

# q = "What is the difference between a process and a program?"
# data = doc_creation(q,"OperatingSystems")

    
    
 