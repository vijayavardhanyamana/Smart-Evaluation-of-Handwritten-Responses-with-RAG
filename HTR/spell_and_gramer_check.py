import language_tool_python
from spellchecker import SpellChecker
  
tool = language_tool_python.LanguageTool('en-US') 

def check_grammar(answer):
  
    my_matches = tool.check(answer)  
    corrected_text = tool.correct(answer)  
    return corrected_text


def correct_spelling(text):
    spell = SpellChecker()
    words = text.split()

    # Find misspelled words
    misspelled = spell.unknown(words)

    # Correct misspelled words
    corrected_text = []
    for word in words:
        if word in misspelled:
            correction = spell.correction(word)
#             print(f"Misspelled word: {word}, Correction: {correction}")  
            corrected_text.append(correction)
        else:
            corrected_text.append(word)
            
    return " ".join(map(str, corrected_text))

def spell_grammer(text):
    spell_check_text = correct_spelling(text)
    
    corrected_text = check_grammar(spell_check_text)
    
    return corrected_text