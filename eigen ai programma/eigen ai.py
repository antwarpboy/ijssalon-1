import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import spacy
import sympy
import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wikipediaapi
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QMessageBox
from PySide6.QtCore import Qt



# Laden van het standaardmodel van spaCy voor Nederlands
nlp = spacy.load("nl_core_news_sm")

def load_dataset():
    # Laden van de Iris-dataset
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    
    # Laden van tekst van een Wikipedia-pagina
    wiki = wikipediaapi.Wikipedia(user_agent='mijn-ai-app/1.0')
    try:
        page = wiki.page('Onderwerp_naar_keuze')
        text = page.text
    except wikipediaapi.exceptions.PageError as e:
        print("Fout bij het ophalen van Wikipedia-pagina:", e)
        text = None
    
    # Laad de AG News-dataset
    ag_news_data = pd.read_csv("D:/vscodedatasets/datasets/agnews.csv")
    
    return X_iris, y_iris, text, ag_news_data

# Call the load_dataset() function to initialize datasets
iris, X_iris, y_iris, text, ag_news_data = load_dataset()
# Gebruik de iris dataset
print(iris.feature_names)
column_names = list(iris.feature_names) + ['target']
iris_df = pd.DataFrame(data=np.c_[X_iris, y_iris], columns=column_names)


X = iris_df.drop(columns=['target'])  # Definieer X in plaats van X_iris
y_iris = iris_df['target']
# Gebruik de AG News datase
ag_news_data = pd.read_csv("D:/vscodedatasets/datasets/agnews.csv")
# Gebruik de tekst van Wikipedia
# Controleren of de tekst is geladen
if text is not None:
    # Doe hier verdere verwerking of analyse met de variabele text
    print("Lengte van de Wikipedia-tekst:", len(text))
    print("Eerste 100 tekens van de Wikipedia-tekst:", text[:100])
else:
    print("Geen tekst gevonden van de Wikipedia-pagina.")


# Voorbeeld van het gebruik van scipy en sympy
# scipy voor numerieke berekeningen
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 7, 11])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
print("Slope:", slope)
print("Intercept:", intercept)

# sympy voor symbolische berekeningen
x = sympy.Symbol('x')
y = sympy.sin(x) + sympy.cos(x)
print("Derivative:", sympy.diff(y, x))

# Definieer een voorbeeldzin
sentence = "SpaCy is een krachtige bibliotheek voor natuurlijke taalverwerking."

# Verwerk de zin met spaCy
doc = nlp(sentence)

# Tokenisatie en POS-tagging
print("Token\t\tPOS-tag")
print("---------------------------")
for token in doc:
    print(f"{token.text}\t\t{token.pos_}")

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
y_test_array = np.array(y_test)
y_test_tensor = torch.tensor(y_test_array, dtype=torch.int64)



# Model bouwen
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 14)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(14, 14)
        self.fc3 = nn.Linear(14, 14)
        self.fc4 = nn.Linear(14, 14)
        self.fc5 = nn.Linear(14, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        return out


input_size = X_train.shape[1]
hidden_size = 11
num_classes = len(np.unique(y_train))
model = SimpleNN(input_size, hidden_size, num_classes)

# Loss en optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

num_epochs = 62
batch_size = 64
for epoch in range(num_epochs):
    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size]

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        optimizer.zero_grad()
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
torch.save(model.state_dict(), 'model.pth')
 

# Laden van het getrainde model
def load_model():
    model = SimpleNN(input_size, hidden_size, num_classes)  # Creëer een instantie van je modelklasse
    model.load_state_dict(torch.load('model.pth'))  # Laad de parameters van het opgeslagen model
    return model


def ask_question(iris):  # Voeg iris als parameter toe
    question = window.question_entry.text()
    if question:
        answer_index = get_answer(question, model, iris)
        answer = iris.target_names[answer_index]
        QMessageBox.information(window, "Antwoord", f"Het voorspelde antwoord is: {answer}")
    else:
        QMessageBox.warning(window, "Fout", "Voer een vraag in.")
        window = MainWindow()
        window.show()
        app.exec_()


def get_answer(question, model, iris):  # Voeg iris als parameter toe
    question_vec = preprocess_question(question)
    question_tensor = torch.tensor(question_vec, dtype=torch.float32)
    outputs = model(question_tensor)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Communicator")
        self.setGeometry(100, 100, 400, 200)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.question_label = QLabel("Vraag:")
        self.question_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.question_label)

        self.question_entry = QLineEdit()
        self.layout.addWidget(self.question_entry)

        self.ask_button = QPushButton("Stel vraag")
        self.ask_button.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 5px;")
        self.ask_button.clicked.connect(self.ask_question)
        self.layout.addWidget(self.ask_button)

        self.answer_label = QLabel("")
        self.answer_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.answer_label)

        self.setLayout(self.layout)

def ask_question(iris):  # Voeg iris als parameter toe
    question = window.question_entry.text()
    if question:
        answer_index = get_answer(question, model, iris)
        answer = iris.target_names[answer_index]
        QMessageBox.information(window, "Antwoord", f"Het voorspelde antwoord is: {answer}")
    else:
        QMessageBox.warning(window, "Fout", "Voer een vraag in.")

def preprocess_question(question):
    # Verwerk de vraag met behulp van spaCy
    doc = nlp(question)
    # Geef een lijst van woordvectoren terug
    return [token.vector for token in doc]

def get_answer(question):
    # Preprocess de vraag
    question_vec = preprocess_question(question)
    # Omzet de vraag in een PyTorch tensor
    question_tensor = torch.tensor(question_vec, dtype=torch.float32)
    # Voer de vraag door het model en verkrijg de voorspelling
    outputs = model(question_tensor)
    _, predicted = torch.max(outputs, 1)
    # Geef het voorspelde antwoord terug
    return predicted.item()

app = QApplication([])
window = MainWindow()
window.show()
app.exec()