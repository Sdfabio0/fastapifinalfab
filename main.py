import uvicorn
from fastapi import FastAPI , Depends
from dotenv import load_dotenv
import os
from typing import Dict
from pydantic import BaseModel
from spacy.lang.en import English
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pickle
from deep_translator import GoogleTranslator
load_dotenv()
nltk.download('punkt')
vars_check = ['classification_code','nom_periode_quart_FR','Nom','classification_description_FR']
sentences = []
filename3 = 'constraints.txt'
f = open(filename3)
with open(filename3, encoding='utf-8') as f:
    for line in f:
        line = nltk.word_tokenize(line)
        sentences.append(line)
f.close()

def get_sql(query):
  input_text = "translate English to SQL: %s </s>" % query
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'],

               attention_mask=features['attention_mask'])

  return tokenizer.decode(output[0])
def create_patterns(dic):
    patterns = []
    for key, values in dic.items():
        if key == 'nom_periode_quart_FR' or key == 'classification_description_FR':
            values = [x.lower() for x in values]
            values = [translator.translate(val.lower(), src='fr', dest='en') for val in values]
        lower_words = [{"LOWER": value} for value in values]
        pattern_line = {"label":key, "pattern":lower_words}
        patterns.append(pattern_line)
    return patterns
def match_constraints(order, sents = sentences, vars_check = vars_check):
    max_match = 0
    val = False
    sents_match = []
    for sent in sents:
        doc = nlp(' '.join(sent))
        lab = [ent.label_ for ent in doc.ents]
        labels = set(lab) & set(vars_check)
        keys = set(order.keys())
        match = labels & keys
        match_len = len(match)
        if max_match == match_len:
            sents_match.append(sent)
        elif max_match < match_len:
            max_match = match_len
            sents_match = []
            sents_match.append(sent)
    return sents_match

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")

translator = GoogleTranslator()

a_file = open('datatable.pkl', "rb")
dic = pickle.load(a_file)
a_file.close()
nlp = English()
ruler = nlp.add_pipe("entity_ruler")
patterns = create_patterns(dic)
ruler.add_patterns(patterns)

# load environment variables
port = os.environ["PORT"]
# initialize FastAPI
app = FastAPI()
@app.get("/")
async def index():
    return {"data": "Hello there only that 68866"}

@app.get("/constraint")
async def read_root(
    num_employees: int = 8,
    class_code: str = 'CASTM',
    classification_code: str = "BARRE",
    nom_periode_quart_FR: str =  "JOUR",
    Nom: str = "MISSISSAUGA EXPRESS",
    classification_description_FR: str = "Prepose aux barres",
    vars_check = vars_check):

    col1 = []
    col2 = []
    col3 = []
    col4 = []
    col5 = []
    col6 = []
    CONDS = []
    list_sent = []
    shift_time = nom_periode_quart_FR
    order = {
  "classification_code": classification_code,
  "nom_periode_quart_FR": nom_periode_quart_FR,
  "Nom": Nom,
  "classification_description_FR": classification_description_FR
}
    match_sentences = match_constraints(order)
    if len(match_sentences)>0:
        for sent in match_sentences:
             CONDS.append(get_sql(' '.join(sent)))
             list_sent.append(' '.join(sent))
    sql_rule = """
    SELECT {}
    FROM EMPLOYEES
    WITH {} = {} AND {} = {}
    """.format(num_employees,'nom_periode_quart_FR',shift_time,'CODE',class_code)
    col1.append(num_employees)
    col2.append(shift_time)
    col3.append(class_code)
    col4.append(sql_rule)
    col5.append(list_sent)
    col6.append(CONDS)
    dic = {'NUM_EMPLOYEES': col1, 'SHIFT_TIME': col2, 'CODE': col3, 'RULE': col4, 'CONSTRAINT_SENTS': col5,'CONDITIONS': col6}
    return dic

@app.get("/sql")
async def generate_sql(sent):
    return get_sql(sent)

if __name__ == "__main__":
    uvicorn.run("main:app", host = '0.0.0.0', port=port, reload=False)