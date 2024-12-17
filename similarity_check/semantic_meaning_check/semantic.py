from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
import gensim.downloader as api
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
import numpy as np
import nltk
from gensim import corpora
from gensim.models import FastText
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex


similarity_model_path = "models/similarity_model"

# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# tokenizer.save_pretrained(similarity_model_path)

# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# model.save(similarity_model_path)

similarity_tokenizer = AutoTokenizer.from_pretrained(similarity_model_path)
similarity_model = SentenceTransformer(similarity_model_path)



path = "models/fasttext_wiki_news.model"

# fasttext = api.load('fasttext-wiki-news-subwords-300')
# fasttext.save(path)

fasttext = KeyedVectors.load(path)

# nltk.download('punkt')
# nltk.download('stopwords')



def similarity_model_score(correct_answer1,correct_answer2,answer):
    correct_answer1_embedding = similarity_model.encode(correct_answer1, convert_to_tensor=True)
    correct_answer2_embedding = similarity_model.encode(correct_answer2, convert_to_tensor=True)
    answer_embedding = similarity_model.encode(answer, convert_to_tensor=True)
    
    cosine_scores1 = util.pytorch_cos_sim(correct_answer1_embedding, answer_embedding)
    cosine_scores2 = util.pytorch_cos_sim(correct_answer2_embedding, answer_embedding)
    
    max_cosine_score = max(cosine_scores1,cosine_scores2)
    
    return max_cosine_score



def preprocess(sentence):
    
    # Lowercase and remove punctuation
    sentence = sentence.lower()
    # Tokenize
    words = word_tokenize(sentence)
    # Remove stop words
    words = [word for word in words if word not in stopwords.words('english')]
    return words

def sentence_to_vec(tokens, model):
    # Filter words that are in the Word2Vec vocabulary
    valid_words = [word for word in tokens if word in model]

    # If there are no valid words, return a zero vector
    if not valid_words:
        return np.zeros(model.vector_size)

    # Compute the average vector
    word_vectors = [model[word] for word in valid_words]
    sentence_vector = np.mean(word_vectors, axis=0)

    return sentence_vector



def compute_scm(tokens1, tokens2, model):
    dictionary = corpora.Dictionary([tokens1, tokens2])
    tokens1 = dictionary.doc2bow(tokens1)
    tokens2 = dictionary.doc2bow(tokens2)
    termsim_index = WordEmbeddingSimilarityIndex(model)
    termsim_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)
    similarity = termsim_matrix.inner_product(tokens1, tokens2, normalized=(True, True))
    return similarity

def fasttext_similarity(correct_answer1,correct_answer2,answer):
    preprocess_correct_answer1 = preprocess(correct_answer1)
    preprocess_correct_answer2 = preprocess(correct_answer2)
    preprocess_answer   = preprocess(answer)
    
    soft_cosine1 = compute_scm(preprocess_correct_answer1, preprocess_answer,fasttext)
    soft_cosine2 = compute_scm(preprocess_correct_answer2, preprocess_answer,fasttext)
    
    return max(soft_cosine1,soft_cosine2)

    





    