import tokenizer
with open("dataset.txt", "r") as f:
    reading = f.read()
    tokens = tokenizer.encoder(reading)
    context_size = 4
    inputs = []
    targets = []

    for i in range(len(tokens) - context_size):
        input_sequence = tokens[i:i + context_size]
        target = tokens[i + context_size]

        inputs.append(input_sequence)
        targets.append(target)
"""print("Inputs:", inputs)
print("Targets:", targets)
"""

