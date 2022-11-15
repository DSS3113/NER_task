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


def get_entities(data):
    entities = set()
    for sent in sent_tokenize(data):
        ner_results = ner(sent)
        for result in ner_results:
            word = str(result['word'])
            if len(word) < 4 or '#' in word:
                continue
            entities.add(word)
    return tuple(entities)
        
class NamedEntities(Resource):
    def post(self):
        data = request.get_json()
        entities = get_entities(data['body_text'])
        body_html = data['body_html']
        for entity in entities:
            body_html = body_html.replace(f" {entity} ", f" <span style='background-color: yellow'>{entity}</span> ")\
                .replace(f" {entity}.", f" <span style='background-color: yellow'>{entity}</span>.")\
                .replace(f" {entity},", f" <span style='background-color: yellow'>{entity}</span>,")\
                .replace(f" {entity}:", f" <span style='background-color: yellow'>{entity}</span>:")\
                .replace(f" {entity}–", f" <span style='background-color: yellow'>{entity}</span>–")\
                .replace(f" {entity}\n", f" <span style='background-color: yellow'>{entity}</span>\n")
        return {'body_html': body_html}

api.add_resource(NamedEntities, "/")

if __name__ == '__main__':
    app.run(debug=True)
