import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from collections import Counter
import string

def remove_stopwords(sentence):
    
    # converting into words
    words = word_tokenize(sentence)

    # Get the set of English stop words
    stop_words = set(stopwords.words('english'))

    # Remove stop words from the list of words
    filtered_words = [word for word in words if word.lower() not in stop_words]
    
    words = [word.lower() for word in words if word.isalpha() and len(word)>1]
    
    return words

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms


def process_sentence(words):
 
    # Find synonyms for each word
    synonym_map = {}
    for word in words:
        synonyms = get_synonyms(word)
        synonyms.add(word)  # Ensure the word itself is included if no synonyms are found
        synonym_map[word] = list(synonyms)
    
    return synonym_map

def tf(dict1):
    no_of_terms_in_document = len(dict1)
    word_frequency = {}
    for i in dict1:
        count = 0
        for j in dict1:
            if i in dict1[j]:
                count+=1
        word_frequency[i] = count
#     print(word_frequency)
    
    for i in word_frequency:
        word_frequency[i] = word_frequency[i]/no_of_terms_in_document
        
    return word_frequency
        
def idf(dict1,dict2):
    no_of_documents = 2
    new_dict = {}
    for i in dict1:
        if i in new_dict:
            continue
        new_dict[i] = 1
        for j in dict2:
            if i in dict2[j]:
                new_dict[i] += 1
                break
    
    for i in dict2:
        if i in new_dict:
            continue
        new_dict[i] = 1
        for j in dict1:
            if i in dict1[j]:
                new_dict[i] += 1
                break
    
    for i in new_dict:
        new_dict[i] /= no_of_documents
    
    return new_dict

def total_tf_idf_value(tf_idf_word_values,synonyms_words):
    value = 0
    for i in synonyms_words:
        for j in synonyms_words[i]:
            if j in tf_idf_word_values:
                value += tf_idf_word_values[j]
                break
    return value

def create_tfidf_values(correct_answer1,correct_answer2):
    correct_answer1_words = remove_stopwords(correct_answer1)
    correct_answer2_words = remove_stopwords(correct_answer2)
    
    correct_synonyms_words1 = process_sentence(correct_answer1_words)
    correct_synonyms_words2 = process_sentence(correct_answer2_words)
    
    tf_correct_answer1 = tf(correct_synonyms_words1)
    tf_correct_answer2 = tf(correct_synonyms_words2)
    
    idf_values = idf(correct_synonyms_words1,correct_synonyms_words2)
    
    tf_idf_word_values = {}

    for i in correct_synonyms_words1:
        value = tf_correct_answer1[i]*idf_values[i]
        if i in tf_idf_word_values:
            tf_idf_word_values[i] = max(tf_idf_word_values[i],value)
        else:
            tf_idf_word_values[i] = value
            
    for i in correct_synonyms_words2:
        value = tf_correct_answer2[i]*idf_values[i]
        if i in tf_idf_word_values:
            tf_idf_word_values[i] = max(tf_idf_word_values[i],value)
        else:
            tf_idf_word_values[i] = value
            
    for i in tf_idf_word_values:
        tf_idf_word_values[i] =  round(tf_idf_word_values[i], 3)
            
    
    total_tfidf_correct_ans_1 = total_tf_idf_value(tf_idf_word_values,correct_synonyms_words1)
    # total_tfidf_correct_ans_1 /= len(correct_answer1_words)
    total_tfidf_correct_ans_2 = total_tf_idf_value(tf_idf_word_values,correct_synonyms_words2)  
    # total_tfidf_correct_ans_2 /= len(correct_answer2_words)
    max_tfidf = min(total_tfidf_correct_ans_1 , total_tfidf_correct_ans_2)
    
    
    return  tf_idf_word_values,max_tfidf

    
    
    
def tfidf_answer_score(answer,tf_idf_word_values,max_tfidf,marks=10):
    answer = remove_stopwords(answer)
    answer_synonyms_words = process_sentence(answer)
    value = total_tf_idf_value(tf_idf_word_values,answer_synonyms_words)
    # print("tfidf value of answer: ",value, "  ,  " "minimum tfidf value of correct answer answer: " ,max_tfidf)
    score = (value/max_tfidf)*marks
    # print(score)
    if score>10:
        return 10
    else:
        return score