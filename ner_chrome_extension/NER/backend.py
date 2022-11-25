# ML imports
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from nltk.tokenize import sent_tokenize

# Flask imports
from flask import Flask, request
from flask_restful import Api, Resource
from flask_cors import CORS

app= Flask(__name__) 
CORS(app)
api = Api(app)

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
ner = pipeline("ner", model=model, tokenizer=tokenizer)
colors = {
    'PER': 'red',
    'ORG': 'orange',
    'LOC': 'yellow',
    'MISC': 'green'
}

def get_entities(data):
    entities = set()
    for sent in sent_tokenize(data):
        ner_results = ner(sent)
        for result in ner_results:
            word = str(result['word'])
            entity_type = str(result['entity'][result['entity'].find('-')+1:])
            if len(word) < 4 or '#' in word:
                continue
            entities.add((word, entity_type))
    return tuple(entities)


"""
PER: red
ORG: orange
LOC: yellow
MISC: green
"""
class NamedEntities(Resource):
    def post(self):
        data = request.get_json()
        entities = get_entities(data['body_text'])
        body_html = data['body_html']
        for entity in entities:
            word = entity[0]
            color = colors[entity[1]]
            body_html = body_html.replace(f" {entity[0]} ", f" <span style='background-color: {color}'>{word}</span> ")\
                .replace(f" {word}.", f" <span style='background-color: {color}'>{word}</span>.")\
                .replace(f" {word},", f" <span style='background-color: {color}'>{word}</span>,")\
                .replace(f" {word}:", f" <span style='background-color: {color}'>{word}</span>:")\
                .replace(f" {word}–", f" <span style='background-color: {color}'>{word}</span>–")\
                .replace(f" {word}\n", f" <span style='background-color: {color}'>{word}</span>\n")
        return {'body_html': body_html}

api.add_resource(NamedEntities, "/")

if __name__ == '__main__':
    app.run(debug=True)
