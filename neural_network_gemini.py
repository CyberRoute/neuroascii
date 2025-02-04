import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1. Data Preparation
# Define your character set and logo

char_to_index = {
    ' ': 0,  # Space
    '⣿': 1,  # Dense block
    '⣷': 2,  # Block upper left
    '⣾': 3,  # Block upper left & right
    '⠉': 4,  # Character from logo
    '⠁': 5,
    '⣻': 6,
    '⣤': 7,
    '⣴': 8,
    '⠟': 9,
    '⠻': 10,
    '⣇': 11,
    '⣀': 12,
    '⡀': 13,
    '⢸': 14,
    '⡇': 15,
    '⢿': 16,
    '⡿': 17,
    # Add any other characters that appear in your logo
}

# Add the new character '⣠' and increment num_chars
char_to_index['⣠'] = len(char_to_index)  # Assign it a new index

index_to_char = {index: char for char, index in char_to_index.items()}  # Rebuild index_to_char after modification
num_chars = len(char_to_index)  # Update the number of chars

logo = """
⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⠁⠀⠀⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣷⣤⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⠉⠉⠉⣿⣿⣿⠉⠉⠉⠟⠉⠉⠉⠻⣿⣿⣿⣿
⣿⣿⣿⣿⠀⠀⠀⣿⣿⣿⠀⠀⠀⣠⣤⡀⠀⠀⢹⣿⣿⣿
⣿⣿⣿⣿⠀⠀⠀⣿⣿⣿⠀⠀⠀⣿⣿⡇⠀⠀⢸⣿⣿⣿
⣿⣿⣿⣿⠀⠀⠀⣿⣿⣿⠀⠀⠀⣿⣿⡇⠀⠀⢸⣿⣿⣿
⣿⣿⣿⣿⣀⣀⣀⣿⣿⣿⣀⣀⣀⣿⣿⣇⣀⣀⣼⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿
"""

# No need to replace '⣠' now since we've added it to the dict.
# logo = logo.replace('⣠', ' ')

data = [[char_to_index[char] for char in line] for line in logo.splitlines() if line]
flattened_data = [char for line in data for char in line]  # Flatten the 2D list

# Sequence Creation
sequence_length = 20  # Adjust this!

X = []
y = []
for i in range(0, len(flattened_data) - sequence_length):
    seq_in = flattened_data[i:i + sequence_length]
    seq_out = flattened_data[i + sequence_length]
    X.append(seq_in)
    y.append(seq_out)

# Reshape and Normalize
X = np.reshape(X, (len(X), sequence_length, 1))
X = X / float(num_chars)  # Normalize input values
y = to_categorical(y, num_classes=num_chars) # One-Hot Encode output

# 2. Model Definition
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))  # Experiment with more units
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dropout(0.3))
model.add(Dense(num_chars, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

# 3. Training
model.fit(X, y, epochs=40, batch_size=64)  # Adjust epochs and batch size

# 4. Generation

def generate_ascii_art(model, seed_sequence, length, width, char_to_index, index_to_char):
    """Generates ASCII art using the trained model."""
    generated_text = seed_sequence.copy()
    for _ in range(length):
        # Prepare the input
        x = np.reshape(generated_text[-sequence_length:], (1, sequence_length, 1))
        x = x / float(len(char_to_index))

        # Predict the next character probabilities
        probabilities = model.predict(x, verbose=0)[0]

        # Sample from the distribution (more creative)
        predicted_index = np.random.choice(range(len(probabilities)), p=probabilities)

        generated_text.append(predicted_index)

    # Reshape to lines:
    num_lines = (len(generated_text) // width)
    reshaped_output = generated_text[:num_lines * width] # Take only full rows

    # Convert to ASCII Art
    ascii_art = ""
    for i in range(0, len(reshaped_output), width):
        line = "".join([index_to_char[index] for index in reshaped_output[i:i+width]])
        ascii_art += line + "\n"

    return ascii_art

# Example Usage
start_index = np.random.randint(0, len(X)-1)
seed_sequence = list(np.reshape(X[start_index], (sequence_length,)).astype(int))  # Convert to list of integers

generated_art = generate_ascii_art(model, seed_sequence, 500, len(data[0]), char_to_index, index_to_char)
print(generated_art)
