import numpy as np
import pandas as pd
import time
import yfinance as yf
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras import Dense
from alpha_vantage.timeseries import TimeSeries
from flask import render_template
from flask_socketio import SocketIO, emit
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Vul hier je eigen API-sleutel in
api_key = 'YOUR_API_KEY'

# Definieer de URL's voor de verschillende symbolen
urls = [
    "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=STOXX50&interval=5min&apikey=" + api_key,
    "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=NDX&interval=5min&apikey=" + api_key,
    "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=XAUUSD&interval=5min&apikey=" + api_key,
    "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=BTCUSD&interval=5min&apikey=" + api_key,
    "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=ETHUSD&interval=5min&apikey=" + api_key,
    "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=EURUSD&interval=5min&apikey=" + api_key
]

# Maak een HTTP-verzoek voor elke URL en verwerk de respons
for url in urls:
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()  # Converteer de respons naar JSON-formaat
        # Verwerk de gegevens zoals nodig
        print(data)  # Bijvoorbeeld: print de gegevens naar de console
    else:
        print("Fout bij het ophalen van gegevens voor:", url)

# Maak een TimeSeries object voor aandelengegevens
timeseries = TimeSeries(key=api_key, output_format='pandas')

# Functie om gegevens op te halen voor een specifiek aandeel
def fetch_data(symbol):
    data, _ = timeseries.get_daily(symbol=symbol, outputsize='full')
    return data

# Functie om technische indicatoren te berekenen
# Functie om de MACD-indicator te berekenen
def calculate_macd(close_price, window_short=12, window_long=26, window_signal=9):
    ema_short = close_price.ewm(span=window_short, min_periods=window_short, adjust=False).mean()
    ema_long = close_price.ewm(span=window_long, min_periods=window_long, adjust=False).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=window_signal, min_periods=window_signal, adjust=False).mean()
    return macd, signal_line
def calculate_technical_indicators(data):
    data['SMA_50'] = data['4. close'].rolling(window=50).mean()
    data['SMA_200'] = data['4. close'].rolling(window=200).mean()
    data['RSI'] = calculate_rsi(data['4. close'])
    data['MACD'], data['Signal_Line'] = calculate_macd(data['4. close'])  # Voeg MACD-berekening toe
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
    X = data[['SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal_Line']].values  # Voeg MACD toe aan de features
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    
    # Train het model
    model = train_model(X_train, y_train)
    
    # Genereer handelssignalen
    signals = generate_signals(data, model)
    
    # Voer handel uit op basis van signalen
    execute_trade(symbol, signals)

# Start geautomatiseerd handelen voor elk opgegeven aandeel
symbols = ['STOXX50', 'NDX', 'XAUUSD', 'BTCUSD', 'ETHUSD','EURUSD']
while True:
    for symbol in symbols:
        auto_trade(symbol)
    time.sleep(86400)  # Wacht 1 dag tussen elke iteratie

# Aanvullende functionaliteit voor het weergeven van knoppen en balansinformatie
def start_trading(event):
    global trading_active
    trading_active = True
    print("Trading is gestart.")

def stop_trading(event):
    global trading_active
    trading_active = False
    print("Trading is gestopt.")

def update_balance():
    global trading_balance, available_balance, total_balance
    trading_balance = np.random.randint(1000, 5000)
    available_balance = np.random.randint(1000, 5000)
    total_balance = trading_balance + available_balance

# Creëer een grafiek om de balansinformatie weer te geven
def plot_balance():
    plt.figure(figsize=(8, 6))
    plt.bar(['Trading Balance', 'Available Balance', 'Total Balance'], [trading_balance, available_balance, total_balance], color=['blue', 'green', 'red'])
    plt.title('Balance Information')
    plt.xlabel('Account Type')
    plt.ylabel('Balance')
    plt.show()

# Creëer knoppen om het handelen te starten en te stoppen
start_button_ax = plt.axes([0.1, 0.05, 0.2, 0.075])
start_button = Button(start_button_ax, 'Start Trading')
start_button.on_clicked(start_trading)

stop_button_ax = plt.axes([0.7, 0.05, 0.2, 0.075])
stop_button = Button(stop_button_ax, 'Stop Trading')
stop_button.on_clicked(stop_trading)

# Initialisatie van variabelen voor balansinformatie
trading_balance = 0
available_balance = 0
total_balance = 0

# Update balansinformatie en plot
update_balance()
plot_balance()
