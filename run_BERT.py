# BertForSequenceClassification => BERT with a language head
# BertTokenizer = a BERT tokenizer (word-piece embedding + positional embedding)
from transformers import BertForSequenceClassification, BertTokenizer

# Tokenizer model for the ignore case version of BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# multi_label_classification Language Head of size 18
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    problem_type="multi_label_classification",
    num_labels=18,
    output_attentions=True,
    output_hidden_states=True
)

inputs = tokenizer("The cat saw the dog", return_tensors="pt")
outputs = model(**inputs)
probs = outputs.logits.softmax(dim=-1)
print(inputs)
print(probs)

"""
You should probably TRAIN this model on a down-stream task to be able 
to use it for predictions and inference.

- Note: the 18 probabilities are randomized

inputs = [ 101, 1996, 4937, 2387, 1996, 3899,  102]
[CLS] The cat saw the dog [SEP]

- the ** unpacks/extracts the values of the 
- dictionary and puts the values from the "input" variable to the keys of the function we're calling!
"""