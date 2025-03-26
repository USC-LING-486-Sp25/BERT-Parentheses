# BertForSequenceClassification => BERT with a language head
# BertTokenizer = a BERT tokenizer (word-piece embedding + positional embedding)
from transformers import BertForSequenceClassification, BertTokenizer
import time

# Tokenizer model for the ignore case version of BERT
t1_tokenizer = time.time()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
t2_tokenizer = time.time()

# multi_label_classification Language Head of size 18
t1_model = time.time()
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    problem_type="multi_label_classification",
    num_labels=18,
    output_attentions=True,
    output_hidden_states=True
)
t2_model = time.time()

t1_inference = time.time()
inputs = tokenizer("The cat saw the dog", return_tensors="pt")
outputs = model(**inputs)
probs = outputs.logits.softmax(dim=-1)
t2_inference = time.time()

print(inputs)
# print(outputs)
print(probs)
print(model)


print(f"Time to load tokenizer = {t2_tokenizer-t1_tokenizer:.2f}")
print(f"Time to load model = {t2_model-t1_model:.2f}")
print(f"Time for inference = {t2_inference-t1_inference:.2f}")
"""
You should probably TRAIN this model on a down-stream task to be able 
to use it for predictions and inference.

- Note: the 18 probabilities are randomized

inputs = [ 101, 1996, 4937, 2387, 1996, 3899,  102]
[CLS] The cat saw the dog [SEP]

- the ** unpacks/extracts the values of the dictionary (inputs is a dictionary!)
- dictionary and puts the values from the "input" variable to the keys of the function we're calling!
"""

"""
{
'input_ids': tensor([[101, 1996, 4937, 2387, 1996, 3899,  102]]), # Token Embeddings
'token_type_ids': tensor( # Sentence 1 gets 0's, sentence 2 gets 1's
    [[0, 0, 0, 0, 0, 0, 0]]
), 
'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]]) # You can mask out a specific word using 0 or 1
    # Fill in the blank Cloze tests (Gesalt like learning)
    # Test are we correcting word in the same topic? what about same part of speech, what about a match for a phrase?
}
"""

"""
Note the "callable object" is like a C functor
"""