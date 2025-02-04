import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Exact target pattern
TARGET_ART = """
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

def preprocess_target_art(art):
    """
    Precisely convert ASCII art to a numerical matrix.
    Each cell represents the specific Braille character's density.
    """
    # Remove newlines and whitespace
    art = art.strip().replace('\n', '')
    
    # Create a matrix representing the art with more nuanced encoding
    matrix = np.zeros((11, 11), dtype=np.float32)
    braille_density = {
        '⠀': 0.0,    # Empty
        '⣾': 0.8,    # Partial fill
        '⣿': 1.0,    # Full fill
        '⣷': 0.9,    # Near full fill
        '⣻': 0.7,    # Partial fill
        '⢿': 0.6,    # Partial fill
        '⠁': 0.2,    # Minimal fill
        '⠀': 0.0,    # Empty
        '⣤': 0.5,    # Half fill
        '⣴': 0.6,    # Partial fill
    }
    
    for i in range(11):
        for j in range(11):
            char = art[i*11 + j]
            matrix[i, j] = braille_density.get(char, 0.0)
    
    return matrix

def create_advanced_dataset(num_samples=2000):
    """Create a more sophisticated training dataset."""
    # Preprocess the target art
    target = preprocess_target_art(TARGET_ART)
    
    # Create input data with structured noise
    X_train = np.random.normal(0.5, 0.2, (num_samples, 11, 11, 1))
    
    # Add some structured variations close to the target
    for i in range(num_samples):
        noise = np.random.normal(0, 0.1, (11, 11))
        X_train[i] += noise
        X_train[i] = np.clip(X_train[i], 0, 1)
    
    Y_train = np.array([target for _ in range(num_samples)])
    
    # Reshape for CNN
    Y_train = Y_train.reshape(-1, 11, 11, 1)
    
    return X_train, Y_train

def create_deep_model():
    """Create a more complex neural network."""
    model = keras.Sequential([
        # Input layer with more complexity
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(11, 11, 1)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Deeper network with residual connections
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Residual block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Add()([
            layers.Conv2D(64, (3, 3), activation='relu', padding='same')(
                layers.Conv2D(64, (3, 3), activation='relu', padding='same')(
                    layers.Conv2D(64, (3, 3), activation='relu', padding='same')(
                        layers.InputLayer(input_shape=(11, 11, 1))(None)
                    )
                )
            ),
            layers.InputLayer(input_shape=(11, 11, 1))(None)
        ]),
        
        # Final layers
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ])
    
    # Compile with more sophisticated optimization
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='mean_squared_error',
        metrics=['accuracy']
    )
    
    return model

def convert_to_braille(matrix):
    """
    Convert matrix to Braille with more nuanced mapping.
    Uses different Braille characters based on density.
    """
    braille_map = [
        '⠀',  # 0.0
        '⠁',  # 0.1-0.2
        '⣤',  # 0.3-0.4
        '⣴',  # 0.5-0.6
        '⣻',  # 0.7-0.8
        '⣾',  # 0.9
        '⣿'   # 1.0
    ]
    
    # Normalize matrix to 0-1 range
    matrix_norm = (matrix - matrix.min()) / (matrix.max() - matrix.min())
    
    # Convert to Braille
    art_lines = []
    for row in matrix_norm:
        line = ''.join(braille_map[min(int(val * len(braille_map)), len(braille_map)-1)] for val in row)
        art_lines.append(line)
    
    return '\n'.join(art_lines)

def train_and_generate():
    # Prepare advanced dataset
    X_train, Y_train = create_advanced_dataset()
    
    # Create and train deep model
    model = create_deep_model()
    history = model.fit(
        X_train, 
        Y_train,
        epochs=100,  # Increased epochs
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )
    
    # Generate multiple attempts
    for attempt in range(3):
        # Random seed input
        test_input = np.random.normal(0.5, 0.2, (1, 11, 11, 1))
        generated = model.predict(test_input)[0, :, :, 0]
        
        print(f"\nGenerated ASCII Art (Attempt {attempt + 1}):")
        print(convert_to_braille(generated))
    
    # Print training metrics
    print("\nTraining Metrics:")
    print(f"Final Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Loss: {history.history['loss'][-1]:.4f}")
    
    return model, history

# Run the training and generation
if __name__ == "__main__":
    model, history = train_and_generate()
