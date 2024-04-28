import openai
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import tkinter as tk
import speech_recognition as sr
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Verbinden met de OpenAI API (vervang 'API_KEY' door je eigen API-sleutel)
openai.api_key = 'API_KEY'

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
optimizer = optim.Adam(model.parameters(), lr=0.003)

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

# Gebruik GPT-3 om een vraag te beantwoorden
def get_gpt3_response(prompt):
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0.7,
        max_tokens=50
    )
    return response.choices[0].text.strip()

question = "Wat is het hoofdbestanddeel van water?"
gpt3_answer = get_gpt3_response(question)
print("GPT-3 antwoord:", gpt3_answer)


# GUI met spraakopnamen of schrijf functie
class ChatGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Chat Interface")

        # Chatvenster
        self.chat_history = tk.Text(self.root, state='disabled')
        self.chat_history.pack(expand=True, fill='both')

        # Invoerveld
        self.input_field = tk.Entry(self.root)
        self.input_field.pack(fill='x')
        self.input_field.bind("<Return>", self.handle_text_input)

        # Spraakherkenning initialiseren
        self.recognizer = sr.Recognizer()

        # Knop voor spraakopname
        self.record_button = tk.Button(self.root, text="Opnemen", command=self.record_speech)
        self.record_button.pack()

        self.root.mainloop()

    def handle_text_input(self, event):
        user_input = self.input_field.get()
        self.input_field.delete(0, 'end')
        self.update_chat_history(f"Jij: {user_input}")
        self.process_input(user_input)

    def record_speech(self):
        try:
            with sr.Microphone() as source:
                print("Luisteren...")
                audio = self.recognizer.listen(source)
                text = self.recognizer.recognize_google(audio)
                self.update_chat_history(f"Jij (gesproken): {text}")
                self.process_input(text)
        except sr.UnknownValueError:
            print("Kon geen spraak herkennen")

        # Verwerk de tekstinvoer en voer het model uit
        # Voeg hier de logica toe om het AI-model te gebruiken
        # en het antwoord aan de chatgeschiedenis toe te voegen       

    def process_input(self, input_text):
     outputs = model(X_test_tensor)
     _, predicted = torch.max(outputs,1)
     predicted_class = predicted[0].item()
     prediction_text = iris.target_names[predicted_class]
     self.update_chat_history(f"ai:voorspelde bloemklasse - {prediction_text}")

    
     def update_chat_history(self, message):
        self.chat_history.configure(state='normal')
        self.chat_history.insert('end', message + '\n')
        self.chat_history.configure(state='disabled')
        self.chat_history.see('end')

if __name__ == "__main__":
    chat_gui = ChatGUI()
