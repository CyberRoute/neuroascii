import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define ASCII characters for rendering
def ascii_art(matrix):
    chars = "@%#*+=-:. "  # Dark to light
    matrix = np.clip(matrix, 0, 1)
    scaled = matrix * (len(chars) - 1)
    return "\n".join("".join(chars[int(val)] for val in row) for row in scaled)

# Define the target logo
def generate_target_logo(size=11):
    logo = np.zeros((size, size))
    
    # Borders
    logo[0, :] = 0.5  # Top border
    logo[-1, :] = 0.5  # Bottom border
    logo[:, 0] = 0.5  # Left border
    logo[:, -1] = 0.5  # Right border
    
    # Diagonal patterns
    for i in range(size):
        logo[i, i] = 1.0  # Main diagonal
        logo[i, size - 1 - i] = 1.0  # Anti-diagonal
    
    # Central block
    logo[3:8, 3:8] = 0.8  # Central square
    
    return logo

# Generate training data
def generate_training_data(num_samples=5000, size=11):
    X_train = []
    Y_train = []
    
    target_logo = generate_target_logo(size)
    
    for _ in range(num_samples):
        # Create structured noise
        noise = np.random.rand(size, size) * 0.3
        noise = np.abs(noise + np.random.randn(size, size) * 0.1)
        
        # Add transformations
        if np.random.rand() > 0.5:
            noise = np.fliplr(noise)
        if np.random.rand() > 0.5:
            noise = np.rot90(noise, k=np.random.randint(1, 4))
        
        X_train.append(noise.reshape(size, size, 1))
        Y_train.append(target_logo.reshape(size, size, 1))
    
    return np.array(X_train), np.array(Y_train)

# Corrected U-Net-like model architecture
def build_model(input_shape=(11, 11, 1)):
    inputs = keras.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)  # 11x11 -> 5x5
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 5x5 -> 2x2
    
    # Bottleneck
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    
    # Decoder
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)  # 2x2 -> 4x4
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)  # 4x4 -> 8x8
    
    # Final upsampling to restore 11x11
    x = layers.UpSampling2D(size=(2, 2))(x)  # 8x8 -> 16x16
    x = layers.Cropping2D(cropping=((2, 3), (2, 3)))(x)  # 16x16 -> 11x11
    
    # Output
    outputs = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
    
    model = keras.Model(inputs, outputs)
    return model

# Prepare data
X_train, Y_train = generate_training_data()

# Build and compile model
model = build_model()
model.compile(optimizer="adam", loss="binary_crossentropy")

# Train the model
model.fit(X_train, Y_train, batch_size=32, epochs=50, validation_split=0.2)

# Generate output
test_input = np.random.rand(1, 11, 11, 1)
predicted_logo = model.predict(test_input)[0, :, :, 0]

# Display result
print("Generated Logo:")
print(ascii_art(predicted_logo))
