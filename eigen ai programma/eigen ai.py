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
import multiprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sortedcontainers import SortedDict
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QMessageBox
from PySide6.QtCore import Qt

def task(iterations_remaining):
    parallel_task(iterations_remaining)

def parallel_task(iterations_remaining):
    # Basisgeval: stop de recursie als het aantal resterende iteraties nul is
    if iterations_remaining == 0:
        return

    # Voer de taken uit voor deze iteratie
    print(f"Parallelle taak wordt uitgevoerd, resterende iteraties: {iterations_remaining}")

    # Maak een pool van processen
    with multiprocessing.Pool(processes=2) as pool:
        # Start processen
        pool.map(task, [iterations_remaining - 1, iterations_remaining - 1])

if __name__ == "__main__":
    initial_iterations = 1  # Stel het aantal iteraties in
    parallel_task(initial_iterations)

# Laden van het standaardmodel van spaCy voor Nederlands
nlp = spacy.load("nl_core_news_sm")

# Functie voor tokenisatie
def tokenize_text(text):
    # Verwerk de tekst met spaCy
    doc = nlp(text)
    # Haal de tokens op en retourneer ze als een lijst van woorden
    return [token.text for token in doc]

# Functie voor lemmatisering
def lemmatize_text(text):
    # Verwerk de tekst met spaCy
    doc = nlp(text)
    # Haal de lemma's op en retourneer ze als een lijst van woorden
    return [token.lemma_ for token in doc]

# Functie voor stopwoordverwijdering
def remove_stopwords(text):
    # Verwerk de tekst met spaCy
    doc = nlp(text)
    # Haal de tokens op en verwijder de stopwoorden
    tokens_without_stopwords = [token.text for token in doc if not token.is_stop]
    # Retourneer de tekst zonder stopwoorden
    return " ".join(tokens_without_stopwords)

# Functie voor zinsontleding
def parse_sentence(text):
    # Verwerk de tekst met spaCy
    doc = nlp(text)
    # Voor elke zin in de tekst
    for sentence in doc.sents:
        # Print de zin en de afzonderlijke delen ervan
        print("Zin:", sentence.text)
        print("Tokens:")
        for token in sentence:
            print(f"{token.text}\t{token.lemma_}\t{token.pos_}\t{token.dep_}")
        print("")

# Voorbeeldtekst
text = "Dit is een voorbeeldzin voor tekstverwerking met spaCy. Het demonstreert tokenisatie, lemmatisering, stopwoordverwijdering en zinsontleding."

# Tokenisatie
tokens = tokenize_text(text)
print("Tokenisatie:")
print(tokens)
print("")

# Lemmatisering
lemmas = lemmatize_text(text)
print("Lemmatisering:")
print(lemmas)
print("")

# Stopwoordverwijdering
text_without_stopwords = remove_stopwords(text)
print("Tekst zonder stopwoorden:")
print(text_without_stopwords)
print("")

# Zinsontleding
print("Zinsontleding:")
parse_sentence(text)

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
    print(ag_news_data.head())                                                             # Print de eerste paar rijen van de dataset

# Gebruik van de STEM-corpusdataset
if wiki_stem_corpus_data is not None:
    print(wiki_stem_corpus_data.head())                                                    # Print de eerste paar rijen van de dataset

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

def perform_numeric_computation():
# scipy voor numerieke berekeningen
   x = np.array([1, 2, 3, 4, 5])
   y = np.array([2, 3, 5, 7, 11])
   slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
   print("Slope:", slope)
   print("Intercept:", intercept)

def perform_symbolic_computation():
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
class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
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

input_size = X_train_standard.shape[1]                                                             # Gebruik de juiste X_train tensor
print("Shape van X_train_standard:", X_train_standard.shape)
hidden_size = 11                                                                                   # die zijn de verborgen lagen  
num_classes = len(np.unique(y_train))
num_layers = 5
model = CustomRNN(input_size, hidden_size, num_classes, num_layers)
# Loss en optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

num_epochs = 62
batch_size = 64
for epoch in range(num_epochs):
    for i in range(0, len(X_train_standard_tensor), batch_size):
        X_batch = X_train_standard_tensor[i:i+batch_size]                                           # Gebruik de juiste X_train tensor
        y_batch = y_train_tensor[i:i+batch_size]                                                    # Gebruik de juiste y_train tensor
        
        
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Functie voor het uitvoeren van grid search {dit dient voor het automatisch optimaliseren van de batch_size, epochs, verborgen lagen, num_lagen en de leersnelheid}
def grid_search(X_train, y_train, X_test, y_test, input_size, num_classes, num_layers):
    best_accuracy = 0
    best_parameters = {}

    learning_rates = [0.001, 0.005, 0.01]
    hidden_sizes = [8, 11, 14]
    num_epochs = [40, 50, 60, 70, 80, 90, 100]
    batch_sizes = [32, 64, 128, 256]
    num_layers_list = [2, 3, 4, 5]
    
    for lr in learning_rates:
        for hidden_size in hidden_sizes:
            for num_epoch in num_epochs:
                for batch_size in batch_sizes:
                    for num_layers_loop in num_layers_list:
                        model = CustomRNN(input_size, hidden_size, num_classes, num_layers_loop)

                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                    
                    for epoch in range(num_epoch):
                        for i in range(0, len(X_train), batch_size):
                            X_batch = X_train[i:i+batch_size]
                            y_batch = y_train[i:i+batch_size]
                            
                            outputs = model(X_batch)
                            loss = criterion(outputs, y_batch)
                            
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    
                    with torch.no_grad():
                        outputs = model(X_test)
                        _, predicted = torch.max(outputs, 1)
                        accuracy = (predicted == y_test).sum().item() / len(y_test)
                        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_parameters = {'learning_rate': lr, 'hidden_size': hidden_size, 'num_epochs': num_epoch, 'batch_size': batch_size, 'num_layers': num_layers}
    
    return best_parameters, best_accuracy
num_layers = 5
# Uitvoeren van grid search
best_params, best_accuracy = grid_search(X_train_standard_tensor, y_train_tensor, X_test_standard_tensor, y_test_tensor, input_size, num_classes, num_layers)


print("Beste hyperparameters gevonden:")
print(best_params)
print("Beste test nauwkeurigheid gevonden:", best_accuracy)

def train_and_test_accuracy_history(model, criterion, optimizer, X_train, y_train, X_test, y_test, num_epochs=100, batch_size=64):
    train_accuracy_history = []
    test_accuracy_history = []

    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_total = 0
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == y_batch).sum().item()
            train_total += y_batch.size(0)
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        train_accuracy = train_correct / train_total
        train_accuracy_history.append(train_accuracy)

        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            test_correct = (predicted == y_test).sum().item()
            test_accuracy = test_correct / len(y_test)
            test_accuracy_history.append(test_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

    return train_accuracy_history, test_accuracy_history

# Evaluatie van het model op de testset
with torch.no_grad():
    outputs = model(X_test_standard_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Test accuracy: {accuracy:.2f}')

# Opslaan van het model
torch.save(model.state_dict(), 'model.pth')

# Stel lists in om nauwkeurigheidsgeschiedenis bij te houden
train_accuracy_history = []
test_accuracy_history = []

# Trainingslus
for epoch in range(num_epochs):
    # Training code hier...
    # Bereken train_accuracy en test_accuracy na elke epoch
    train_accuracy = ...  # Code om train_accuracy te berekenen
    test_accuracy = ...   # Code om test_accuracy te berekenen
    
    # Voeg de nauwkeurigheden toe aan de lijsten
    train_accuracy_history.append(train_accuracy)
    test_accuracy_history.append(test_accuracy)


# Voor het plotten van de nauwkeurigheidsgrafiek
def plot_accuracy(train_accuracy_history, test_accuracy_history):
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracy_history, label='Train accuracy')
    plt.plot(test_accuracy_history, label='Test accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.show()

# Voor het berekenen en plotten van de confusion matrix
def plot_confusion_matrix(y_test, predicted, classes):
    cm = confusion_matrix(y_test, predicted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plt.show()

# Controleren of de geschiedenis van nauwkeurigheid bestaat
if 'train_accuracy_history' in globals() and 'test_accuracy_history' in globals():
    plot_accuracy(train_accuracy_history, test_accuracy_history)
else:
    print("Train en test nauwkeurigheid geschiedenis zijn niet beschikbaar.")

# Controleren of de test labels en voorspellingen bestaan
if 'y_test_tensor' in globals() and 'predicted' in globals():
    classes = iris.target_names  # Definieer je klassenlabels, gebruikmakend van de iris dataset
    plot_confusion_matrix(y_test_tensor, predicted, classes)
else:
    print("Test labels of voorspellingen zijn niet beschikbaar.")

def train_model():
    train_accuracy_history = []
    test_accuracy_history = []
    
    # Trainingslus hier...
    
    return train_accuracy_history, test_accuracy_history

train_accuracy_history, test_accuracy_history = train_model()
plot_accuracy(train_accuracy_history, test_accuracy_history)    
 
''' hier word het getrainde model opgeladen '''
# Laden van het getrainde model
def load_model():
    model = CustomRNN(input_size, hidden_size, num_classes)                                          # Creëer een instantie van je modelklasse
    model.load_state_dict(torch.load('model.pth'))                                                   # Laad de parameters van het opgeslagen model
    return model

""" vanaf hier verwerken we de vragen en verkrijgen de antwoorden op onze vragen."""

class Iris:
    def __init__(self):
        iris_dataset = load_iris()
        self.target_names = iris_dataset.target_names
        self.data = iris_dataset.data                                                                 # Laad de Iris-data om er voor te zorgen dat de AI de vraag verstaat 

def ask_question(iris, max_length):
    question = window.question_entry.text()
    if question:
        if "STEM-corpus" in question:
            answer = "Je hebt een vraag gesteld over de STEM-corpus."
        elif "Wikipedia" in question:
            topic = question.split("Wikipedia ")[-1]
            text = load_wikipedia_text(topic)
            if text:
                answer = f"Tekst van Wikipedia over '{topic}': {text[:100]}..."
            else:
                answer = "Er is geen tekst gevonden van de Wikipedia-pagina."
        elif "scipy" in question:
            perform_numeric_computation()
            answer = "Numerieke berekening uitgevoerd met scipy."
        elif "sympy" in question:
            perform_symbolic_computation()
            answer = "Symbolische berekening uitgevoerd met sympy."
        else:
            answer_index = get_answer(question, model, iris, max_length)
            answer = iris.target_names[answer_index]
        window.answer_label.setText(f"Het voorspelde antwoord is: {answer}")
    else:
        QMessageBox.warning(window, "Fout", "Voer een vraag in.")

def preprocess_question(question, max_length):
    if "STEM-corpus" in question:
        # Vraag over STEM-corpus
        stem_corpus_text = load_wiki_stem_corpus_dataset()
        if stem_corpus_text is not None:
            doc = nlp(stem_corpus_text)
            vectors = [token.vector for token in doc]
            num_features = len(vectors[0])
            if len(vectors) < max_length:
                vectors += [np.zeros(num_features)] * (max_length - len(vectors))
            elif len(vectors) > max_length:
                vectors = vectors[:max_length]
            print("Lengte van de vectors:", len(vectors))
            print("Vorm van de eerste vector:", vectors[0].shape)
            return vectors
        else:
            print("Fout: STEM-corpus kon niet worden geladen.")
            return None
    elif "Wikipedia" in question:
        # Vraag over Wikipedia-tekst
        topic = question.split("Wikipedia")[1].strip()
        wikipedia_text = load_wikipedia_text(topic)
        if wikipedia_text is not None:
            doc = nlp(wikipedia_text)
            vectors = [token.vector for token in doc]
            num_features = len(vectors[0])
            if len(vectors) < max_length:
                vectors += [np.zeros(num_features)] * (max_length - len(vectors))
            elif len(vectors) > max_length:
                vectors = vectors[:max_length]
            print("Lengte van de vectors:", len(vectors))
            print("Vorm van de eerste vector:", vectors[0].shape)
            return vectors
        else:
            print("Fout: Wikipedia-tekst kon niet worden geladen.")
            return None
    elif "scipy" in question:
        # Vraag over scipy numerieke berekeningen
        perform_numeric_computation()
        return None
    elif "sympy" in question:
        # Vraag over sympy symbolische berekeningen
        perform_symbolic_computation()
        return None
    else:
        # Normale vraag, verwerk zoals eerder
        doc = nlp(question)
        vectors = [token.vector for token in doc]
        num_features = len(vectors[0])
        if len(vectors) < max_length:
            vectors += [np.zeros(num_features)] * (max_length - len(vectors))
        elif len(vectors) > max_length:
            vectors = vectors[:max_length]
        print("Lengte van de vectors:", len(vectors))
        print("Vorm van de eerste vector:", vectors[0].shape)
        return vectors
       

def get_answer(question, model, iris, max_length):
    if "STEM-corpus" in question:
        return 0                                                                            # Terugkeer van een fictieve index voor de STEM-corpus
    elif "Wikipedia" in question:
        return 1                                                                            # Terugkeer van een fictieve index voor Wikipedia
    elif "scipy" in question:
        return 2                                                                            # Terugkeer van een fictieve index voor scipy
    elif "sympy" in question:
        return 3                                                                            # Terugkeer van een fictieve index voor sympy
    else:
        question_vec = preprocess_question(question, max_length)
        question_vec_np = np.array(question_vec).flatten()
        combined_features = np.concatenate((question_vec_np, iris.data.flatten()))
        combined_tensor = torch.tensor([combined_features], dtype=torch.float32)
        outputs = model(combined_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()
    
# GUI met PySide6
class MainWindow(QMainWindow):
    def __init__(self, iris, max_length):
        super().__init__()
        self.iris = iris                                                                       # Opslaan van de iris_dataset als attribuut
        self.max_length = max_length                                                           # Opslaan van de max_length als een attribuut
        self.setWindowTitle("mijn ai vriendje")                                                # Instellen van de titel van het venster
        self.setGeometry(100, 100, 600, 700)                                                   # Instellen van de grootte en positie van het venster
        
        self.central_widget = QWidget()                                                        # Het centrale widget maken
        self.setCentralWidget(self.central_widget)                                             # Het centrale widget instellen

        self.layout = QVBoxLayout(self.central_widget)                                         # Het verticale layout maken

        self.question_label = QLabel("Vraag:")                                                 # Label voor de vraag maken
        self.layout.addWidget(self.question_label)                                             # Label toevoegen aan het layout

        self.question_entry = QLineEdit()                                                      # Textveld voor de vraag maken
        self.layout.addWidget(self.question_entry)                                             # Textveld toevoegen aan het layout

        self.ask_button = QPushButton("Stel vraag")                                            # Knop om de vraag te stellen maken
        self.ask_button.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 5px;")
        self.ask_button.clicked.connect(self.ask_question)                                     # Verbinden van de knop met een functie
        self.layout.addWidget(self.ask_button)                                                 # Knop toevoegen aan het layout

        self.answer_label = QLabel("")                                                         # Label voor het antwoord maken
        self.answer_label.setAlignment(Qt.AlignCenter)                                         # Het antwoord centreren
        self.layout.addWidget(self.answer_label)                                               # Label toevoegen aan het layout

        # Knoppen voor numerieke en symbolische berekeningen
        self.numeric_button = QPushButton("Voer numerieke berekening uit")                     # Knop voor numerieke berekeningen maken
        self.numeric_button.clicked.connect(self.perform_numeric_computation)                  # Verbinden van de knop met een functie
        self.layout.addWidget(self.numeric_button)                                             # Knop toevoegen aan het layout

        self.symbolic_button = QPushButton("Voer symbolische berekening uit")                  # Knop voor symbolische berekeningen maken
        self.symbolic_button.clicked.connect(self.perform_symbolic_computation)                # Verbinden van de knop met een functie
        self.layout.addWidget(self.symbolic_button)                                            # Knop toevoegen aan het layout

        self.setLayout(self.layout)                                                            # Het layout instellen voor het venster


    def ask_question(self):
        question = self.question_entry.text()                                                   # Haal de vraagtekst op uit de gebruikersinvoer
        question_vec = preprocess_question(question, self.max_length)                           # Preprocess de vraag
        question_vec_np = np.array(question_vec).flatten()                                      # Maak een 1D-array van de vraagvectoren
        combined_features = np.concatenate((question_vec_np, self.iris.data.flatten()))         # Combineer vraag- en iris-kenmerken
        combined_tensor = torch.tensor([combined_features], dtype=torch.float32)                # Converteer naar een PyTorch-tensor
        outputs = model(combined_tensor)                                                        # Voer het model uit op de gecombineerde kenmerken
        _, predicted = torch.max(outputs, 1)                                                    # Bepaal de index van de voorspelde klasse
        predicted_class = self.iris.target_names[predicted.item()]                              # Bepaal de naam van de voorspelde klasse
        self.answer_label.setText(f"Het voorspelde antwoord is: {predicted_class}")             # Toon het voorspelde antwoord in het antwoordlabel


    def perform_numeric_computation(self):
        x = np.array([1, 2, 3, 4, 5])                                                           # Voorbeeldgegevens voor numerieke berekening
        y = np.array([2, 3, 5, 7, 11])
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)              # Voer lineaire regressie uit met scipy
        result = f"Slope: {slope}, Intercept: {intercept}"                                      # Genereer het resultaat
        self.answer_label.setText(result)                                                       # Geef het resultaat weer in het antwoordlabel van de GUI

    def perform_symbolic_computation(self):
        x = sympy.Symbol('x')                                                                   # Definieer een symbool 'x' voor symbolische berekening
        y = sympy.sin(x) + sympy.cos(x)                                                         # Definieer een symbolische expressie: sin(x) + cos(x)
        derivative = sympy.diff(y, x)                                                           # Bereken de afgeleide van de symbolische expressie naar 'x'
        result = f"Derivative: {derivative}"                                                    # Genereer het resultaat
        self.answer_label.setText(result)                                                       # Geef het resultaat weer in het antwoordlabel van de GUI
   

app = QApplication([])
iris = Iris()
max_length = 100                                                                                # Je kunt deze waarde aanpassen op basis van je preprocess_question functie
window = MainWindow(iris, max_length)                                                           # Geef max_length door bij het maken van het MainWindow-object
window.show()
app.exec()


