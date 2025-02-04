import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Define the ASCII art logo as a list of strings.
ascii_art = [
    "⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷",
    "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿",
    "⣿⣿⣿⣿⠁⠀⠀⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿",
    "⣿⣿⣿⣿⣷⣤⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿",
    "⣿⣿⣿⣿⠉⠉⠉⣿⣿⣿⠉⠉⠉⠟⠉⠉⠉⠻⣿⣿⣿⣿",
    "⣿⣿⣿⣿⠀⠀⠀⣿⣿⣿⠀⠀⠀⣠⣤⡀⠀⠀⢹⣿⣿⣿",
    "⣿⣿⣿⣿⠀⠀⠀⣿⣿⣿⠀⠀⠀⣿⣿⡇⠀⠀⢸⣿⣿⣿",
    "⣿⣿⣿⣿⠀⠀⠀⣿⣿⣿⠀⠀⠀⣿⣿⡇⠀⠀⢸⣿⣿⣿",
    "⣿⣿⣿⣿⣀⣀⣀⣿⣿⣿⣀⣀⣀⣿⣿⣇⣀⣀⣼⣿⣿⣿",
    "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿",
    "⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿"
]

# 2. Preprocess the art so that every row has the same number of characters.
max_width = max(len(row) for row in ascii_art)
ascii_art = [row.ljust(max_width) for row in ascii_art]  # pad with spaces

n_rows = len(ascii_art)
n_cols = max_width

# 3. Build a mapping from characters to indices.
all_chars = sorted(set("".join(ascii_art)))
char_to_idx = {ch: i for i, ch in enumerate(all_chars)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}
n_classes = len(all_chars)

print("Characters used:", char_to_idx)

# 4. Create the dataset: for every coordinate (row, col) we map to a normalized (x,y) and a target class.
coords = []
targets = []
for i, row in enumerate(ascii_art):
    for j, ch in enumerate(row):
        # Normalize coordinates to [0,1]
        x = i / (n_rows - 1) if n_rows > 1 else 0.0
        y = j / (n_cols - 1) if n_cols > 1 else 0.0
        coords.append([x, y])
        targets.append(char_to_idx[ch])
        
coords = torch.tensor(coords, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.long)

# 5. Define a simple MLP network.
class AsciiMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=n_classes):
        super(AsciiMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

model = AsciiMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# 6. Train the network on the dataset.
n_epochs = 5000
for epoch in range(n_epochs):
    optimizer.zero_grad()
    outputs = model(coords)  # shape: [n_samples, n_classes]
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}] Loss: {loss.item():.4f}")

# 7. After training, “query” the network for every coordinate and reconstruct the ASCII art.
with torch.no_grad():
    # Query for each coordinate in the grid in the same order as our dataset.
    outputs = model(coords)
    predicted = torch.argmax(outputs, dim=1).numpy()

# Reshape the predicted indices back to (n_rows, n_cols)
predicted_grid = predicted.reshape(n_rows, n_cols)

# Convert predicted indices to characters
generated_art = []
for i in range(n_rows):
    row_chars = "".join(idx_to_char[idx] for idx in predicted_grid[i])
    generated_art.append(row_chars)

print("\nGenerated ASCII Art:")
print("\n".join(generated_art))

