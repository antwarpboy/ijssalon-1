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
from sortedcontainers import SortedDict
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QMessageBox
from PySide6.QtCore import Qt


# Laden van het standaardmodel van spaCy voor Nederlands
nlp = spacy.load("nl_core_news_sm")

# Laden van de Iris-dataset
def load_iris_dataset():
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    return iris, X_iris, y_iris

# Laden van de AG News-dataset
def load_ag_news_dataset():
    try:
        ag_news_data = pd.read_csv("D:/vscodedatasets/datasets/agnews.csv")
        print("AG News dataset succesvol geladen.")
        return ag_news_data
    except FileNotFoundError:
        print("Fout: Het opgegeven pad naar de AG News dataset is ongeldig.")
        return None

# Laden van de STEM-corpusdataset
def load_wiki_stem_corpus_dataset():
    try:
        wiki_stem_corpus_data = pd.read_csv("D:/vscodedatasets/datasets/wiki_stem_corpus.csv")
        print("STEM-corpusdataset succesvol geladen.")
        return wiki_stem_corpus_data
    except FileNotFoundError:
        print("Fout: Het opgegeven pad naar de STEM-corpusdataset is ongeldig.")
        return None

# Laden van tekst van een Wikipedia-pagina
def load_wikipedia_text(topic):
    wiki = wikipediaapi.Wikipedia(user_agent='mijn-ai-app/1.0')
    try:
        page = wiki.page(topic)
        text = page.text
        return text
    except wikipediaapi.exceptions.PageError as e:
        print("Fout bij het ophalen van Wikipedia-pagina:", e)
        return None

# Functie om datasets te laden en te retourneren
def load_datasets():
    iris, X_iris, y_iris = load_iris_dataset()
    X = X_iris  # Definieer X als de kenmerken van de Iris-dataset
    ag_news_data = load_ag_news_dataset()
    wiki_stem_corpus_data = load_wiki_stem_corpus_dataset()
    return iris, X_iris, y_iris, X, ag_news_data, wiki_stem_corpus_data


# Functie om een B-tree-index voor de Iris-dataset te maken
def index_iris_dataset(X_iris):
    iris_index = SortedDict()
    for i, features in enumerate(X_iris):
        iris_index[i] = features
    return iris_index

# Functie voor het opzoeken van kenmerken in de Iris-dataset
def lookup_iris_features(iris_index, index):
    if index in iris_index:
        features = iris_index[index]
        print("Kenmerken van de bloem met index", index, ":", features)
    else:
        print("Index niet gevonden in de Iris-dataset.")

# Laden van datasets en initialisatie
iris, X_iris, y_iris, X, ag_news_data, wiki_stem_corpus_data = load_datasets()

print("iris data shape (features):", iris['data'].shape)

# Voorbeeld van het gebruik van de functies
iris_index = index_iris_dataset(X_iris)
lookup_iris_features(iris_index, 0)

# Gebruik van de AG News-dataset
if ag_news_data is not None:
    print(ag_news_data.head())  # Print de eerste paar rijen van de dataset

# Gebruik van de STEM-corpusdataset
if wiki_stem_corpus_data is not None:
    print(wiki_stem_corpus_data.head())  # Print de eerste paar rijen van de dataset

# Laden van tekst van een Wikipedia-pagina
wikipedia_text = load_wikipedia_text("Artificial intelligence")

# Controleren of de tekst is geladen
if wikipedia_text is not None:
    print("Lengte van de Wikipedia-tekst:", len(wikipedia_text))
    print("Eerste 100 tekens van de Wikipedia-tekst:", wikipedia_text[:100])
else:
    print("Geen tekst gevonden van de Wikipedia-pagina.")

'''dit om hogere wiskundige bewerkingen'''

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
''' dit is om spacy te testen'''
# Definieer een voorbeeldzin
sentence = "SpaCy is een krachtige bibliotheek voor natuurlijke taalverwerking."

# Verwerk de zin met spaCy
doc = nlp(sentence)

# Tokenisatie en POS-tagging
print("Token\t\tPOS-tag")
print("---------------------------")
for token in doc:
    print(f"{token.text}\t\t{token.pos_}")

''' testen en trainen van het model met verschillende scalers '''

# Train/test splitsing
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)


# Normalisatie van de gegevens met StandardScaler
standard_scaler = StandardScaler()
X_train_standard_scaled = standard_scaler.fit_transform(X_train)
X_test_standard_scaled = standard_scaler.transform(X_test)

# Definieer X_train_standard na normalisatie
X_train_standard = X_train_standard_scaled

# Definieer input_size
input_size = X_train_standard.shape[1]

# Normalisatie van de gegevens met MinMaxScaler
minmax_scaler = MinMaxScaler()
X_train_minmax_scaled = minmax_scaler.fit_transform(X_train)
X_test_minmax_scaled = minmax_scaler.transform(X_test)

# Normalisatie van de gegevens met RobustScaler
robust_scaler = RobustScaler()
X_train_robust_scaled = robust_scaler.fit_transform(X_train)
X_test_robust_scaled = robust_scaler.transform(X_test)


# Omzetten naar PyTorch-tensors voor X_train_standard en X_test_standard
X_train_standard_tensor = torch.tensor(X_train_standard_scaled, dtype=torch.float32)
X_test_standard_tensor = torch.tensor(X_test_standard_scaled, dtype=torch.float32)

# Omzetten van y_train en y_test naar PyTorch-tensors
y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
y_test_tensor = torch.tensor(y_test, dtype=torch.int64)


''' het eigenlijk ai model bouwen , dit wil zeggen de verschillende lagen en verborgen lagen, de batch grootte , hoeveel epochs en de leersnelheid'''

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

input_size = X_train_standard.shape[1]  # Gebruik de juiste X_train tensor
hidden_size = 11                        # die zijn de verborgen lagen  
num_classes = len(np.unique(y_train))
model = SimpleNN(input_size, hidden_size, num_classes)

# Loss en optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

num_epochs = 62
batch_size = 64
for epoch in range(num_epochs):
    for i in range(0, len(X_train_standard_tensor), batch_size):
        X_batch = X_train_standard_tensor[i:i+batch_size]  # Gebruik de juiste X_train tensor
        y_batch = y_train_tensor[i:i+batch_size]  # Gebruik de juiste y_train tensor

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

''' hier evalueren we het model op de testset'''
# Evaluatie van het model op de testset
with torch.no_grad():
    outputs = model(X_test_standard_tensor)  # Gebruik de juiste X_test tensor
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Test accuracy: {accuracy:.2f}')
torch.save(model.state_dict(), 'model.pth')
 
''' hier word het getrainde model opgeladen '''
# Laden van het getrainde model
def load_model():
    model = SimpleNN(input_size, hidden_size, num_classes)  # Creëer een instantie van je modelklasse
    model.load_state_dict(torch.load('model.pth'))  # Laad de parameters van het opgeslagen model
    return model


# Functies voor het verwerken van vragen en het verkrijgen van antwoorden
def preprocess_question(question, max_length):
    doc = nlp(question)
    vectors = [token.vector for token in doc]
    if len(vectors) < max_length:
        # Pad the question with zero vectors
        vectors += [np.zeros_like(vectors[0])] * (max_length - len(vectors))
    elif len(vectors) > max_length:
        # Truncate the question if it's longer than max_length
        vectors = vectors[:max_length]
    return vectors

def ask_question(iris, max_length):  # Voeg max_length als parameter toe
    question = window.question_entry.text()
    if question:
        answer_index = get_answer(question, model, iris, max_length)  # Pas max_length als parameter toe
        answer = iris.target_names[answer_index]
        QMessageBox.information(window, "Antwoord", f"Het voorspelde antwoord is: {answer}")
    else:
        QMessageBox.warning(window, "Fout", "Voer een vraag in.")

def get_answer(question, model, iris, max_length):  
    question_vec = preprocess_question(question, max_length)  
    question_tensor = torch.tensor([question_vec], dtype=torch.float32)  # Voeg een extra dimensie toe voor de batch
    outputs = model(question_tensor)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

''' dit is de GUI waar we het uitzicht van de app kunnen maken en aanpassen'''

class MainWindow(QMainWindow):
    def __init__(self, iris):
        super().__init__()
        self.setWindowTitle("AI Communicator")
        self.setGeometry(100, 100, 600, 700)  # Vergroot het venster
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.question_label = QLabel("Vraag:")
        self.layout.addWidget(self.question_label)

        self.question_entry = QLineEdit()
        self.layout.addWidget(self.question_entry)

        self.ask_button = QPushButton("Stel vraag")
        self.ask_button.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 5px;")
        # Verbind de knop met de methode ask_question en geef de juiste waarde voor max_length door
        self.ask_button.clicked.connect(lambda: ask_question(iris, len(nlp(self.question_entry.text()).vector)))
        self.layout.addWidget(self.ask_button)

        self.answer_label = QLabel("")  # Voeg een label toe voor het antwoord
        self.answer_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.answer_label)

        self.setLayout(self.layout)

    # Verplaats de ask_question-methode naar de klasse MainWindow
    def ask_question(self):  
        question = self.question_entry.text()
        if question:
            answer_index = get_answer(question, model, iris)
            answer = iris.target_names[answer_index]
            QMessageBox.information(self, "Antwoord", f"Het voorspelde antwoord is: {answer}")
        else:
            QMessageBox.warning(self, "Fout", "Voer een vraag in.")

app = QApplication([])
window = MainWindow(iris)
max_length = X_train_standard.shape[1]  # Lengte van de trainingsgegevens
window.ask_button.clicked.connect(lambda: ask_question(iris, max_length))  # Verbind de knop met de methode ask_question
window.show()
app.exec()
