import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from classes import *  # assuming your NeuralNetwork and Layer classes are in classes.py

# === 1. Load digits dataset ===
digits = load_digits()
X = digits.data              # shape (1797, 64)
y = digits.target.reshape(-1, 1)  # shape (1797, 1)

# === 2. Normalize inputs ===
X = X / 16.0  # original values are in range [0, 16]

# === 3. One-hot encode labels ===
encoder = OneHotEncoder(sparse_output=False)  # works in newer versions
y_encoded = encoder.fit_transform(y)  # shape (1797, 10)

# === 4. Split into train/test ===
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# === 5. Create neural network ===
net = NeuralNetwork([
    Layer(64, 32, 'relu'),
    # Layer(64, 32, 'sigmoid'),
    Layer(32, 10, 'softmax')
])
# === 6. Train the network ===
epochs = 200
lr = 0.001

for epoch in range(epochs):
    total_loss = 0
    for x, y in zip(X_train, y_train):
        loss = net.one_example(x, y)
        total_loss += loss
    net.apply_gradients(lr)

    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch+1}, loss = {total_loss:.4f}")

# === 7. Evaluate ===
correct = 0
for x, y in zip(X_test, y_test):
    predicted_class = net.predict(x)
    true_class = np.argmax(y)
    if predicted_class == true_class:
        correct += 1

accuracy = correct / len(X_test)
print(f"Test accuracy: {accuracy:.4f}")
