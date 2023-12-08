from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Set the paths to the downloaded model files
model_checkpoint = r'E:\arabert'
# model_checkpoint = r'pytorch_model.bin'
model_config = r'E:\arabert\config.json'
tokenizer_vocab = r'E:\arabert'

model_dir = "arabert2"  

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_vocab)
model = AutoModel.from_pretrained(model_checkpoint, config=model_config)

# Input sentences
sentence1 = "بسم الله الرحمن الرحيم"
sentence2 = "الحمد لله رب العالمين"

# Tokenize and obtain embeddings for each sentence
input_ids1 = tokenizer.encode(sentence1, return_tensors="pt")
input_ids2 = tokenizer.encode(sentence2, return_tensors="pt")

with torch.no_grad():
    outputs1 = model(input_ids1)
    outputs2 = model(input_ids2)

# Get the embeddings for the [CLS] token (first token in the sequence)
embedding1 = outputs1.last_hidden_state[:, 0, :]
embedding2 = outputs2.last_hidden_state[:, 0, :]

# Convert embeddings to numpy arrays
embedding1_np = embedding1.cpu().numpy()
embedding2_np = embedding2.cpu().numpy()

# Calculate cosine similarity
similarity_matrix = cosine_similarity(embedding1_np, embedding2_np)

# The similarity_matrix is a 1x1 matrix, as we are comparing two vectors
similarity_percentage = similarity_matrix[0][0] * 100

# Print the similarity percentage
print(f"Similarity: {similarity_percentage:.2f}%")