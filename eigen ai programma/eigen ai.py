import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Laden van de dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train/test splitsing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisatie van de gegevens
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Omzetten naar PyTorch-tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
y_test_tensor = torch.tensor(y_test, dtype=torch.int64)

# Model bouwen
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = X_train.shape[1]
hidden_size = 10
num_classes = len(np.unique(y_train))
model = SimpleNN(input_size, hidden_size, num_classes)

# Loss en optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 60  # Definieer de variabele num_epochs
# Training van het model met batches
batch_size = 32
for epoch in range(num_epochs):
    for i in range(0, len(X_train_tensor), batch_size):
        # Batch selecteren
        X_batch = X_train_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size]

        # Forward pass en loss berekenen
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass en optimizer stap
        optimizer.zero_grad()  # Reset de gradients
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluatie van het model op de testset
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Test accuracy: {accuracy:.2f}')

# laad je getrainde ai-model
# model = load_model()

def get_answer(question):
    # Verwerk de vraag met behulp van je AI-model
    # answer = model.predict(question)
    # Voor dit voorbeeld geven we een hardgecodeerd antwoord
    answer = "Dit is een antwoord op je vraag."
    return answer

def ask_question():
    question = question_entry.get()
    if question:
        answer = get_answer(question)
        messagebox.showinfo("Antwoord", answer)
    else:
        messagebox.showwarning("Fout", "Voer een vraag in.")

# GUI opzetten
root = tk.Tk()
root.title("AI Communicator")    # dit is de naam dat zichbaar is als titel naam

question_label = tk.Label(root, text="Vraag:")
question_label.pack()

question_entry = tk.Entry(root, width=50)
question_entry.pack()

ask_button = tk.Button(root, text="Stel vraag", command=ask_question)
ask_button.pack()

root.mainloop()