with open("dataset.txt", "r") as f:
    reading = f.read()


# Build vocabulary
words = list(set(reading.split()))
words.sort()

vocabulary = {}

for index, word in enumerate(words):
    vocabulary[word] = index


# Build reverse vocabulary
reverse_vocabulary = {}

for word, index in vocabulary.items():
    reverse_vocabulary[index] = word


# Encoder: text → token IDs
def encoder(text):
    text_list = text.split()
    encoded_list = []

    for word in text_list:
        token_id = vocabulary[word]
        encoded_list.append(token_id)

    return encoded_list


# Decoder: token IDs → text
def decoder(token_ids):
    decoded_words = []

    for token_id in token_ids:
        word = reverse_vocabulary[token_id]
        decoded_words.append(word)

    return " ".join(decoded_words)


# Test
"""encoded = encoder("I love Python")
print("Encoded:", encoded)

decoded = decoder(encoded)
print("Decoded:", decoded)
"""

