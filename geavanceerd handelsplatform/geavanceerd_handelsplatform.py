import numpy as np
import pandas as pd
import time
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, LSTM
from alpha_vantage.timeseries import TimeSeries

# Vul hier je eigen API-sleutel in
api_key = 'YOUR_API_KEY'

# Maak een TimeSeries object voor aandelengegevens
timeseries = TimeSeries(key=api_key, output_format='pandas')

# Functie om gegevens op te halen voor een specifiek aandeel
def fetch_data(symbol):
    data, _ = timeseries.get_daily(symbol=symbol, outputsize='full')
    return data

# Functie om technische indicatoren te berekenen
def calculate_technical_indicators(data):
    data['SMA_50'] = data['4. close'].rolling(window=50).mean()
    data['SMA_200'] = data['4. close'].rolling(window=200).mean()
    data['RSI'] = calculate_rsi(data['4. close'])
    return data

# Functie om de RSI-indicator te berekenen
def calculate_rsi(close_price, window=14):
    delta = close_price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Functie om de gegevens voor het model voor te bereiden
def prepare_data(data):
    data = data.dropna()
    X = data[['SMA_50', 'SMA_200', 'RSI']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Functie om het AI-model te trainen
def train_model(X_train, y_train):
    model = Sequential()
    model.add(Dense(units=50, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    return model

# Functie om handelssignalen te genereren
def generate_signals(data, model):
    X = prepare_data(data)
    signals = model.predict(X)
    return signals

# Functie om handel uit te voeren op basis van signalen
def execute_trade(symbol, signals):
    last_signal = signals[-1]
    if last_signal >= 0.5:
        # Voer kooporder uit
        print("Buying {}.".format(symbol))
    else:
        # Voer verkooporder uit
        print("Selling {}.".format(symbol))

# Hoofdfunctie om geautomatiseerd handelen uit te voeren
def auto_trade(symbol):
    # Haal gegevens op voor het aandeel
    data = fetch_data(symbol)
    
    # Bereken technische indicatoren
    data = calculate_technical_indicators(data)
    
    # Split de gegevens in trainings- en testsets
    X = prepare_data(data)
    y = (data['4. close'].shift(-1) > data['4. close']).astype(int).values[:-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train het model
    model = train_model(X_train, y_train)
    
    # Genereer handelssignalen
    signals = generate_signals(data, model)
    
    # Voer handel uit op basis van signalen
    execute_trade(symbol, signals)

# Start geautomatiseerd handelen voor het opgegeven aandeel
symbol = 'AAPL'  # Voorbeeld ticker (Apple)
while True:
    auto_trade(symbol)
    time.sleep(86400)  # Wacht 1 dag tussen elke iteratie
