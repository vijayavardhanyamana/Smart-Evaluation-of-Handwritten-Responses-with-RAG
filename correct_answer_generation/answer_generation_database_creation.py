import os

from correct_answer_generation.create_database import create_database_main 
from correct_answer_generation.related_content_creation import doc_creation
from correct_answer_generation.answer_generation import query_

def database_creation(path):
    create_database_main(path)
    
def answer_generation(path,query):
    # collection_name = os.path.splitext(os.path.basename(path))[0]
    path = path.replace("/", "_")
    data = doc_creation(query,path)
    correct_answers = query_(query,data)
    return correct_answers
    
    
# ans = answer_generation("OperatingSystems","What is the process, and how does it differ from a program?")
# # data = doc_creation(q,"OperatingSystems")

    

