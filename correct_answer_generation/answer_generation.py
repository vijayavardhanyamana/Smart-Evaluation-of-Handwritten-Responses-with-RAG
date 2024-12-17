from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel

save_directory = "models/saved_model"
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModelForSeq2SeqLM.from_pretrained(save_directory)


def query_(query,doc):    
    input_text = f"Given the following context: {doc}, please provide a thorough and detailed answer to the question: {query}. Your response should be structured with sections, cover all aspects mentioned in the context, and contain at least 300 words."

    inputs = tokenizer(input_text, return_tensors="pt")
    
    temperature = 0.8
    top_k = 40
    top_p = 0.98
    max_length = 1200
    repetition_penalty = 1.1
    num_beams = 1
    
    # Generate answer
    outputs = model.generate(
        **inputs,
        do_sample= True,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        num_beams=num_beams    
    )

    answers = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return answers

