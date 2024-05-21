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

# Laden van het standaardmodel van spaCy voor Nederlands
nlp = spacy.load("nl_core_news_sm")

# Voer parallelle taak uit
def parallel_task():
   
    print("Parallelle taak wordt uitgevoerd")

    # Maak processen
    process1 = multiprocessing.Process(target=parallel_task)
    process2 = multiprocessing.Process(target=parallel_task)

    # Start processen
    process1.start()
    process2.start()

    # Wacht op de voltooiing van processen
    process1.join()
    process2.join()

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

# Controleren of een GPU beschikbaar is, anders de CPU gebruiken
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modelparameters
input_size = X_train_standard_tensor.shape[1]
hidden_size = 32  # Definieer de grootte van de verborgen laag
num_classes = len(np.unique(y_train))
num_epochs = 100  # Aantal epochs
batch_size = 64  # Batchgrootte
sequence_length = 10 
# Stel de invoergegevens in (dummygegevens)
X_batch = torch.randn(batch_size, sequence_length, input_size)

# Controleer de vorm van X_batch
print("Vorm van X_batch voor aanpassing:", X_batch.shape)

# Pas de vorm aan naar (batch_size, sequence_length, input_size)
X_batch = X_batch.view(batch_size, sequence_length, input_size)

# Controleer opnieuw de vorm van X_batch
print("Vorm van X_batch na aanpassing:", X_batch.shape)
# Definieer het aantal lagen voor het RNN-model
num_layers = 5  # Stel het aantal lagen in 

# Definieer de lege lijsten voor train_accuracy_history en test_accuracy_history
train_accuracy_history = []
test_accuracy_history = []


# Definieer de klassenlabels
classes = ['class1', 'class2', 'class3']  # Vervang 'class1', 'class2', 'class3' door de daadwerkelijke klassenlabels die je hebt


''' het eigenlijk ai model bouwen , dit wil zeggen de verschillende lagen en verborgen lagen, de batch grootte , hoeveel epochs en de leersnelheid'''

# Model bouwen
class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CustomRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, y=None):
        # Als de invoergegevens niet gebatcht zijn, voegen we een extra dimensie toe om te batchen
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            if y is not None:
                 y = y.unsqueeze(0)
        
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
    
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
    
        # Als doelgegevens aanwezig zijn, moeten we ook hun grootte aanpassen
        if y is not None:
             # Als de doelgegevens een dimensie van 1 hebben, passen we deze aan naar de batchgrootte
          if len(y.shape) == 1:
            y = y.view(batch_size, -1)
    
        return out




# Model initialiseren met alle vereiste argumenten
model = CustomRNN(input_size, hidden_size, num_layers, num_classes)


# Loss en optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Model trainen
for epoch in range(num_epochs):
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    for i in range(0, len(X_train_standard_tensor), batch_size):
        X_batch = X_train_standard_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size]

        # Afhandelen van de laatste batch
        if i + batch_size > len(X_train_standard_tensor):
            batch_size_last = len(X_train_standard_tensor) - i
            X_batch = X_train_standard_tensor[i:i+batch_size_last]
            y_batch = y_train_tensor[i:i+batch_size_last]

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    # Print de loss na elke 10 epochs
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
    num_layers = [2, 3, 4, 5]
    
    for lr in learning_rates:
        for hidden_size in hidden_sizes:
            for num_epoch in num_epochs:
                for batch_size in batch_sizes:
                    model =CustomRNN(input_size, hidden_size, num_classes,num_layers)
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

# Plot nauwkeurigheidsgrafiek
plt.plot(train_accuracy_history, label='Train accuracy')
plt.plot(test_accuracy_history, label='Test accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()
plt.show()

# Bereken en plot de confusion matrix
cm = confusion_matrix(y_test_tensor, predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot()
plt.show()

