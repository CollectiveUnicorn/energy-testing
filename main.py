import random

import numpy as np
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
from scipy.stats import energy_distance
from transformers import GPT2Tokenizer

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pre-trained Word2Vec model
word2vec_model = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin',
    binary=True
)


def get_word_embedding(word):
    try:
        return word2vec_model[word]
    except KeyError:
        return np.zeros(word2vec_model.vector_size)


def get_embeddings(text):
    tokens = tokenizer.tokenize(text)
    embeddings = [get_word_embedding(token) for token in tokens]
    return np.mean(embeddings, axis=0)


def calculate_similarities(text1, text2):
    vector1 = get_embeddings(text1)
    vector2 = get_embeddings(text2)
    energy_dist = 1 - energy_distance(vector1, vector2)
    cosine_sim = 1 - cosine(vector1, vector2)  # Convert cosine distance to similarity
    return energy_dist, cosine_sim


def scramble_text(text):
    words = text.split()
    random.shuffle(words)
    return ' '.join(words)


def shift_text(text, shift):
    return text[shift:] + text[:shift]


def replace_random_chars(text, num_replacements):
    chars = list(text)
    for _ in range(num_replacements):
        index = random.randint(0, len(chars) - 1)
        chars[index] = random.choice('abcdefghijklmnopqrstuvwxyz')
    return ''.join(chars)


def swap_halves(text):
    mid = len(text) // 2
    return text[mid:] + text[:mid]


# Load the original text
with open("test/my_larger_education.txt", "r") as file:
    original_text = file.read()

# Create modified versions
scrambled_text = scramble_text(original_text)
shifted_text = shift_text(original_text, len(original_text) // 10)  # Shift by 10% of text length
replaced_text = replace_random_chars(original_text, len(original_text) // 20)  # Replace 5% of characters
swapped_halves_text = swap_halves(original_text)

# Calculate similarities
versions = {
    "Scrambled": scrambled_text,
    "Shifted": shifted_text,
    "Replaced": replaced_text,
    "Swapped Halves": swapped_halves_text
}

results = {}
for name, text in versions.items():
    energy_sim, cosine_sim = calculate_similarities(original_text, text)
    results[name] = {"Energy": energy_sim, "Cosine": cosine_sim}

# Print results
print("Similarity scores (higher is more similar):")
print("{:<15} {:<20} {:<20}".format("Version", "Energy Similarity", "Cosine Similarity"))
print("-" * 55)
for name, scores in results.items():
    print("{:<15} {:<20.4f} {:<20.4f}".format(name, scores["Energy"], scores["Cosine"]))

# Determine most similar version for each measure
most_similar_energy = max(results, key=lambda x: results[x]["Energy"])
most_similar_cosine = max(results, key=lambda x: results[x]["Cosine"])

print("\nMost similar version (Energy Distance):", most_similar_energy)
print(f"Energy Similarity score: {results[most_similar_energy]['Energy']:.4f}")

print("\nMost similar version (Cosine Similarity):", most_similar_cosine)
print(f"Cosine Similarity score: {results[most_similar_cosine]['Cosine']:.4f}")
