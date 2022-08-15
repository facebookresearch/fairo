from transformers import DistilBertTokenizer, DistilBertModel

import torch

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased', return_dict=True)

inputs = tokenizer("cube", return_tensors="pt")
print(inputs["input_ids"])
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

print(last_hidden_states.size())