# ============================================================
# 1. IMPORTS
# ============================================================

# Tokenizer-এর vocabulary এবং encoder/decoder ব্যবহার করব।
import tokenizer

# Mathematical operations-এর জন্য NumPy।
import numpy as np

# Training data ব্যবহার করব শুধু training-এর সময়।
import training_data

# Model file আছে কি না check করার জন্য।
import os


# ============================================================
# 2. BASIC MODEL SETTINGS
# ============================================================

# Vocabulary-তে মোট কতগুলো token আছে।
vocab_size = len(tokenizer.vocabulary)

# প্রতিটি token-এর embedding vector-এর size।
embedding_size = 8

# যে file-এ trained model save হবে।
model_file = "sunnygpt_model.npz"


# ============================================================
# 3. MODEL PARAMETERS
# ============================================================

# ------------------------------------------------------------
# যদি আগে থেকে trained model থাকে → LOAD
# ------------------------------------------------------------

if os.path.exists(model_file):

    print("Saved model found! Loading model... 😎")

    loaded_model = np.load(model_file)

    embedding = loaded_model["embedding"]
    weights = loaded_model["weights"]
    bias = loaded_model["bias"]

    print("Model loaded successfully! 😎")


# ------------------------------------------------------------
# যদি model না থাকে → নতুন model তৈরি
# ------------------------------------------------------------

else:

    print("No saved model found.")
    print("Creating a new model... 🧠")

    # প্রতিটি token-এর জন্য random embedding।
    embedding = np.random.randn(
        vocab_size,
        embedding_size
    )

    # Hidden vector থেকে vocabulary scores তৈরি করবে।
    weights = np.random.randn(
        embedding_size,
        vocab_size
    )

    # প্রতিটি token-এর জন্য bias।
    bias = np.zeros(vocab_size)


# ============================================================
# 4. SOFTMAX
# ============================================================

# Raw scores → probability distribution।
def softmax(scores):

    # Numerical stability-এর জন্য maximum বাদ দিই।
    exp_scores = np.exp(
        scores - np.max(scores)
    )

    # সব exponential value-এর যোগফল।
    total = np.sum(exp_scores)

    # Probability distribution।
    probabilities = exp_scores / total

    return probabilities


# ============================================================
# 5. FORWARD PASS
# ============================================================

# Input token IDs → model probabilities।
def forward(input_sequence):

    # --------------------------------------------------------
    # Step 1: Embedding lookup
    # --------------------------------------------------------

    embedded_input = embedding[input_sequence]


    # --------------------------------------------------------
    # Step 2: Mean pooling
    # --------------------------------------------------------

    hidden = embedded_input.mean(axis=0)


    # --------------------------------------------------------
    # Step 3: Linear layer
    # --------------------------------------------------------

    scores = hidden @ weights + bias


    # --------------------------------------------------------
    # Step 4: Softmax
    # --------------------------------------------------------

    probabilities = softmax(scores)

    return probabilities


# ============================================================
# 6. CROSS ENTROPY LOSS
# ============================================================

def cross_entropy(probability):

    # খুব ছোট probability-এর ক্ষেত্রে
    # log(0) সমস্যা এড়াতে tiny value ব্যবহার করি।
    probability = max(probability, 1e-12)

    loss = -np.log(probability)

    return loss


# ============================================================
# 7. LOSS CALCULATION
# ============================================================

def calculate_loss(input_sequence, target):

    # Model prediction।
    probabilities = forward(input_sequence)

    # Target token-এর probability।
    target_probability = probabilities[target]

    # Cross entropy loss।
    loss = cross_entropy(target_probability)

    return loss


# ============================================================
# 8. TRAINING
# ============================================================

# ------------------------------------------------------------
# শুধু নতুন model তৈরি হলে training হবে।
# Saved model থাকলে এই পুরো অংশ SKIP হবে।
# ------------------------------------------------------------

if not os.path.exists(model_file):

    print("Training started... 🔥")

    # মোট কতবার training dataset-এর উপর model চলবে।
    epochs = 100

    # Numerical gradient-এর ছোট পরিবর্তন।
    epsilon = 0.0001

    # Weight কতটা পরিবর্তন হবে।
    learning_rate = 0.1


    # ========================================================
    # Epoch loop
    # ========================================================

    for epoch in range(epochs):

        # Dataset-এর প্রতিটি input-target pair।
        for input_sequence, target in zip(
            training_data.inputs,
            training_data.targets
        ):

            # ------------------------------------------------
            # Gradient matrix
            # ------------------------------------------------

            grad_weights = np.zeros_like(weights)


            # ------------------------------------------------
            # প্রতিটি weight-এর gradient বের করি
            # ------------------------------------------------

            for i in range(weights.shape[0]):

                for j in range(weights.shape[1]):

                    # বর্তমান weight সংরক্ষণ।
                    original = weights[i, j]


                    # ----------------------------------------
                    # Weight একটু কমাই
                    # ----------------------------------------

                    weights[i, j] = original - epsilon

                    loss_minus = calculate_loss(
                        input_sequence,
                        target
                    )


                    # ----------------------------------------
                    # Weight একটু বাড়াই
                    # ----------------------------------------

                    weights[i, j] = original + epsilon

                    loss_plus = calculate_loss(
                        input_sequence,
                        target
                    )


                    # ----------------------------------------
                    # Numerical gradient
                    # ----------------------------------------

                    gradient = (
                        loss_plus - loss_minus
                    ) / (2 * epsilon)


                    # Gradient matrix-এ রাখি।
                    grad_weights[i, j] = gradient


                    # আসল weight ফিরিয়ে আনি।
                    weights[i, j] = original


            # ------------------------------------------------
            # Gradient descent
            # ------------------------------------------------

            weights = (
                weights
                - learning_rate * grad_weights
            )


        # ----------------------------------------------------
        # Epoch শেষে loss
        # ----------------------------------------------------

        loss = calculate_loss(
            training_data.inputs[-1],
            training_data.targets[-1]
        )

        # প্রতি 10 epoch-এ শুধু progress দেখাব।
        if (epoch + 1) % 10 == 0:

            print(
                "Epoch:",
                epoch + 1,
                "Loss:",
                loss
            )


    # ========================================================
    # 9. MODEL SAVE
    # ========================================================

    np.savez(
        model_file,
        embedding=embedding,
        weights=weights,
        bias=bias
    )

    print("Model saved successfully! 😎")


# ============================================================
# 10. USER INPUT → TOKENIZER → MODEL → PREDICTION
# ============================================================

print()
print("SunnyGPT is ready! 🤖")


# User-এর কাছ থেকে text নেব।
user_text = input("Enter text: ")


# ------------------------------------------------------------
# Step 1: Text → Token IDs
# ------------------------------------------------------------

input_ids = tokenizer.encoder(user_text)

print("Input IDs:", input_ids)


# ------------------------------------------------------------
# Step 2: Token IDs → Model
# ------------------------------------------------------------

probabilities = forward(input_ids)


# ------------------------------------------------------------
# Step 3: Highest probability token
# ------------------------------------------------------------

predicted_id = np.argmax(probabilities)

print("Predicted ID:", predicted_id)


# ------------------------------------------------------------
# Step 4: Token ID → Word
# ------------------------------------------------------------

predicted_word = tokenizer.decoder(
    [predicted_id]
)

print("Predicted word:", predicted_word)


# ============================================================
# END
# ============================================================
