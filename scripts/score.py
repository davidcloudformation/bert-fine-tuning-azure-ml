import json
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from azureml.core.model import Model

def init():
    global model
    global tokenizer
    model_path = Model.get_model_path('bert-model')
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def run(data):
    try:
        inputs = json.loads(data)
        texts = inputs['texts']
        encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        outputs = model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=1).tolist()
        return json.dumps(predictions)
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})