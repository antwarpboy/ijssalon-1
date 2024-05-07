import numpy as np
import pandas as pd
import time
import requests
import config
import json
import hashlib
import secrets
import sqlite3
import asyncio
import unittest
import yfinance as yf
from PySide6 import QtWidgets, QtGui
from unittest.mock import patch
from io import StringIO
from unittest.mock import patch,MagicMock
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras import Dense, Dropout
from keras import regularizers
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.widgets import Button
from config import api_key,OANDA_API_KEY,OANDA_ACCOUNT_ID
from sklearn.model_selection import GridSearchCV
from keras import KerasClassifier

class User:
    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.salt = secrets.token_hex(16)  # Willekeurige zoutwaarde voor wachtwoordhashing
        self.password_hash = self._hash_password(password)

    def _hash_password(self, password):
        # Functie om het wachtwoord te hashen met behulp van SHA-256 en een unieke zoutwaarde
        hash_input = password.encode('utf-8') + self.salt.encode('utf-8')
        return hashlib.sha256(hash_input).hexdigest()

    def check_password(self, password):
        # Functie om te controleren of het opgegeven wachtwoord overeenkomt met het opgeslagen wachtwoord
        return self.password_hash == self._hash_password(password)

class Database:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.create_users_table()

    def create_users_table(self):
        # Maak een SQLite-tabel voor gebruikersgegevens (username, email, password_hash, salt)
        self.conn.execute('''CREATE TABLE IF NOT EXISTS users (
                            username TEXT PRIMARY KEY,
                            email TEXT,
                            password_hash TEXT,
                            salt TEXT
                            )''')
        self.conn.commit()

    def register_user(self, user):
        # Voeg een nieuwe gebruiker toe aan de database
        self.conn.execute('''INSERT INTO users (username, email, password_hash, salt)
                            VALUES (?, ?, ?, ?)''', (user.username, user.email, user.password_hash, user.salt))
        self.conn.commit()

    def get_user_by_username(self, username):
        # Haal een gebruiker op uit de database op basis van gebruikersnaam
        cursor = self.conn.execute('''SELECT * FROM users WHERE username = ?''', (username,))
        row = cursor.fetchone()
        if row:
            return User(row[0], row[1], row[2], row[3])
        else:
            return None

class AuthService:
    def __init__(self, database):
        self.database = database

    def register(self, username, email, password):
        # Registreer een nieuwe gebruiker
        if self.database.get_user_by_username(username):
            print("Gebruikersnaam is al in gebruik.")
            return False
        else:
            user = User(username, email, password)
            self.database.register_user(user)
            print("Registratie succesvol.")
            return True

    def login(self, username, password):
        # Log in met gebruikersnaam en wachtwoord
        user = self.database.get_user_by_username(username)
        if user and user.check_password(password):
            print("Inloggen succesvol.")
            return True
        else:
            print("Ongeldige gebruikersnaam of wachtwoord.")
            return False

# Voorbeeldgebruik
db = Database('user_database.db')
auth_service = AuthService(db)

# Registratie voorbeeld
auth_service.register('user1', 'user1@example.com', 'password123')

# Inloggen voorbeeld
auth_service.login('user1', 'password123')

# Definieer de PaymentGateway klasse voor betalingsverwerking
class PaymentGateway:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://paymentgatewayapi.com/api"

    def process_payment(self, amount, currency, user_id):
        endpoint = f"{self.base_url}/process_payment"
        payload = {"api_key": self.api_key, "amount": amount, "currency": currency, "user_id": user_id}
        response = requests.post(endpoint, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return True, data.get("transaction_id")
            else:
                return False, data.get("error_message")
        else:
            return False, "Fout bij het verwerken van de betaling"
class ForexBrokerAPI:
    def __init__(self, api_key, account_id):
        self.api_key = api_key
        self.account_id = account_id
        self.base_url = "https://api-fxtrade.oanda.com/v3"

    def place_order(self, symbol, units, side, type='MARKET'):
        endpoint = f"{self.base_url}/accounts/{self.account_id}/orders"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'order': {
                'instrument': symbol,
                'units': units,
                'side': side,
                'type': type
            }
        }
        response = requests.post(endpoint, headers=headers, data=json.dumps(data))
        if response.status_code == 201:
            print("Order placed successfully.")
        else:
            print("Failed to place order:", response.json())

# Haal de API-sleutels op uit het configuratiebestand
oanda_api_key = config.OANDA_API_KEY
oanda_account_id = config.OANDA_ACCOUNT_ID
mijn_eigen_api_sleutel = config.MIJN_EIGEN_API_SLEUTEL

# Controleer of de API-sleutels zijn ingesteld
if oanda_api_key == "" or oanda_account_id == "" or mijn_eigen_api_sleutel == "":
    raise ValueError("Een of meer API-sleutels zijn niet ingesteld in het configuratiebestand")

# Maak een instantie van de ForexBrokerAPI-klasse
broker_api = ForexBrokerAPI(oanda_api_key, oanda_account_id)
def analyze_data(data):
    # Voer hier de analyse uit op de gegeven data
    pass  # Plaats hier de code voor de analyse

# Functie om gegevens op te halen voor een specifiek aandeel
def fetch_data(symbol, function, interval='5min', outputsize='full'):
    """
     Parameters:
    - symbol (str): Het symbool van het aandeel waarvoor gegevens moeten worden opgehaald.
    - function (str): Het soort gegevens dat moet worden opgehaald (bijv. 'TIME_SERIES_INTRADAY').
    - interval (str, optioneel): Het interval voor de gegevens (standaard is '5min').
    - outputsize (str, optioneel): De grootte van de uitvoer (standaard is 'full').
    

    Returns:
    - pd.DataFrame of None: Een DataFrame met de opgehaalde gegevens of None als er een fout optreedt.
    """
    # Controleer of symbol en function strings zijn
    if not isinstance(symbol, str) or not isinstance(function, str):
        raise TypeError("Symbol en function moeten strings zijn.")

    # Controleer of interval en outputsize strings zijn
    if not isinstance(interval, str) or not isinstance(outputsize, str):
        raise TypeError("Interval en outputsize moeten strings zijn.")

    # Controleer of symbol en function niet leeg zijn
    if not symbol or not function:
        raise ValueError("Symbol en function mogen niet leeg zijn.")

    url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Controleer of er een fout is opgetreden bij het ophalen van de gegevens
        data = response.json()
        if 'Time Series' in data:
            return pd.DataFrame(data['Time Series (5min)']).T
        else:
            print("Geen geldige gegevens gevonden voor:", symbol)
            return None
    except requests.exceptions.RequestException as e:
        print("Fout bij het ophalen van gegevens voor:", symbol)
        print("Foutmelding:", e)
        return None
    except json.JSONDecodeError as e:
        print("Fout bij het decoderen van de JSON-response voor:", symbol)
        print("Foutmelding:", e)
        return None

# Functie om technische indicatoren te berekenen
def calculate_bollinger_bands(close_price, window=20, num_std=2):
    rolling_mean = close_price.rolling(window=window).mean()
    rolling_std = close_price.rolling(window=window).std()
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std
    return upper_band, lower_band

def calculate_stochastic_oscillator(close_price, high_price, low_price, window=14):
    lowest_low = low_price.rolling(window=window).min()
    highest_high = high_price.rolling(window=window).max()
    stochastic_oscillator = 100 * (close_price - lowest_low) / (highest_high - lowest_low)
    return stochastic_oscillator
class TestCalculateBollingerBands(unittest.TestCase):
    def test_calculate_bollinger_bands(self):
        # Voorbeeld invoerdata
        close_price = pd.Series([10, 12, 15, 14, 16, 18, 17, 20, 22, 21])

        # Bereken Bollinger Bands
        upper_band, lower_band = calculate_bollinger_bands(close_price)

        # Controleer of de resultaten geldige pandas Series zijn
        self.assertIsInstance(upper_band, pd.Series)
        self.assertIsInstance(lower_band, pd.Series)

class TestCalculateStochasticOscillator(unittest.TestCase):
    def test_calculate_stochastic_oscillator(self):
        # Voorbeeld invoerdata
        close_price = pd.Series([10, 12, 15, 14, 16, 18, 17, 20, 22, 21])
        high_price = pd.Series([11, 13, 16, 15, 17, 19, 18, 21, 23, 22])
        low_price = pd.Series([9, 11, 14, 13, 15, 17, 16, 19, 21, 20])

        # Bereken Stochastic Oscillator
        stochastic_oscillator = calculate_stochastic_oscillator(close_price, high_price, low_price)

        # Controleer of de resultaten geldig zijn
        self.assertIsInstance(stochastic_oscillator, pd.Series)


def calculate_atr(high_price, low_price, close_price, window=14):
    high_low = high_price - low_price
    high_close = np.abs(high_price - close_price.shift())
    low_close = np.abs(low_price - close_price.shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr
class TestCalculateATR(unittest.TestCase):
    def test_calculate_atr(self):
        # Voorbeeld invoerdata
        high_price = pd.Series([11, 13, 16, 15, 17, 19, 18, 21, 23, 22])
        low_price = pd.Series([9, 11, 14, 13, 15, 17, 16, 19, 21, 20])
        close_price = pd.Series([10, 12, 15, 14, 16, 18, 17, 20, 22, 21])

        # Bereken Average True Range (ATR)
        atr = calculate_atr(high_price, low_price, close_price)

        # Controleer of de resultaten geldig zijn
        self.assertIsInstance(atr, pd.Series)

if __name__ == '__main__':
    unittest.main()

def calculate_ema(data, window):
    """
    Bereken het exponentieel gewogen bewegende gemiddelde (EMA) van de gegeven data met het opgegeven venster.

    Parameters:
    - data (pandas.Series): Pandas Series met de gegevens waarvoor het EMA wordt berekend.
    - window (int): Het venster voor het EMA-berekening.

    Returns:
    - ema (pandas.Series): Pandas Series met de EMA-waarden.
    """
    ema = data.ewm(span=window, min_periods=window).mean()
    return ema

def calculate_macd(close_price, window_short=12, window_long=26, window_signal=9):
    """
    Bereken MACD (Moving Average Convergence Divergence), signaallijn en histogram.

    Parameters:
    - close_price (pandas.Series): Pandas Series met sluitingsprijzen van aandelen.
    - window_short (int, optioneel): Het venster voor de korte EMA (standaard is 12).
    - window_long (int, optioneel): Het venster voor de lange EMA (standaard is 26).
    - window_signal (int, optioneel): Het venster voor de signaallijn (standaard is 9).

    Returns:
    - macd (pandas.Series): Pandas Series met MACD-waarden.
    - signal_line (pandas.Series): Pandas Series met signaallijn-waarden.
    - histogram (pandas.Series): Pandas Series met histogram-waarden.
    """
    # Bereken korte en lange EMA
    ema_short = calculate_ema(close_price, window_short)
    ema_long = calculate_ema(close_price, window_long)

    # Bereken MACD-lijn
    macd_line = ema_short - ema_long

    # Bereken signaallijn
    signal_line = calculate_ema(macd_line, window_signal)

    # Bereken histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


class TestCalculateMACD(unittest.TestCase):
    def test_calculate_macd(self):
        # Voorbeeld invoerdata
        close_price = pd.Series([10, 12, 15, 14, 16, 18, 17, 20, 22, 21])

        # Bereken EMA's voor de invoerdata
        ema_short = calculate_ema(close_price, 12)
        ema_long = calculate_ema(close_price, 26)

        # Verwachte MACD en signaallijn
        expected_macd = ema_short - ema_long
        expected_signal_line = calculate_ema(expected_macd, 9)

        # Bereken MACD en signaallijn met de functie
        macd, signal_line = calculate_macd(close_price)

        # Vergelijk de berekende MACD met de verwachte MACD
        self.assertTrue(macd.equals(expected_macd))

        # Vergelijk de berekende signaallijn met de verwachte signaallijn
        self.assertTrue(signal_line.equals(expected_signal_line))

if __name__ == '__main__':
    unittest.main()

def calculate_rsi(close_price, window=14):
    delta = close_price.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
class TestCalculateRSI(unittest.TestCase):
    def test_calculate_rsi(self):
        # Voorbeeld invoerdata
        close_price = pd.Series([10, 12, 15, 14, 16, 18, 17, 20, 22, 21])

        # Verwachte uitvoerdata
        expected_rsi = pd.Series([np.nan, np.nan, 100.0, 88.8889, 100.0, 100.0, 85.1852, 100.0, 100.0, 97.9021])

        # Bereken RSI
        rsi = calculate_rsi(close_price)

        # Vergelijk de berekende RSI met de verwachte RSI
        self.assertTrue(np.allclose(rsi, expected_rsi, equal_nan=True))

if __name__ == '__main__':
    unittest.main()


def calculate_technical_indicators(data):
    data['SMA_50'] = data['4. close'].rolling(window=50).mean()
    data['SMA_200'] = data['4. close'].rolling(window=200).mean()
    data['RSI'] = calculate_rsi(data['4. close'])
    
    # Bereken EMA en voeg toe aan data
    ema_window = 20  # bijvoorbeeld, kies het gewenste venster voor EMA
    data['EMA'] = calculate_ema(data['4. close'], ema_window)
    
    # Bereken MACD en andere indicatoren
    data['MACD'], data['Signal_Line'], _ = calculate_macd(data['4. close'])

    data['Upper_Band'], data['Lower_Band'] = calculate_bollinger_bands(data['4. close'])
    data['Stochastic_Oscillator'] = calculate_stochastic_oscillator(data['4. close'], data['2. high'], data['3. low'])
    data['ATR'] = calculate_atr(data['2. high'], data['3. low'], data['4. close'])
    
    return data


class TestCalculateTechnicalIndicators(unittest.TestCase):
    def test_calculate_technical_indicators(self):
        # Voorbeeld invoerdata
        data = pd.DataFrame({
            '4. close': [10, 12, 15, 14, 16, 18, 17, 20, 22, 21],
            '2. high': [11, 13, 16, 15, 17, 19, 18, 21, 23, 22],
            '3. low': [9, 11, 14, 13, 15, 17, 16, 19, 21, 20]
        })

        # Verwachte uitvoerdata
        expected_sma_50 = pd.Series([np.nan, np.nan, np.nan, np.nan, 13.8, 14.4, 15.0, 15.6, 16.2, 16.8])
        expected_sma_200 = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 16.6])
        expected_rsi = pd.Series([np.nan, np.nan, 100.0, 88.8889, 100.0, 100.0, 85.1852, 100.0, 100.0, 97.9021])
        expected_upper_band, expected_lower_band = calculate_bollinger_bands(data['4. close'])
        expected_stochastic_oscillator = calculate_stochastic_oscillator(data['4. close'], data['2. high'], data['3. low'])
        expected_atr = calculate_atr(data['2. high'], data['3. low'], data['4. close'])
        expected_ema_window = 20  # venster voor EMA
        expected_ema = calculate_ema(data['4. close'], expected_ema_window)
        expected_macd, expected_macd_signal, expected_macd_histogram = calculate_macd(data['4. close'])
        expected_bollinger_upper = expected_upper_band  # Bollinger Bovenste Band is gelijk aan de bovenste band berekend door `calculate_bollinger_bands`
        expected_bollinger_lower = expected_lower_band  # Bollinger Onderste Band is gelijk aan de onderste band berekend door `calculate_bollinger_bands`

        # Bereken technische indicatoren
        result = calculate_technical_indicators(data)

        # Vergelijk de berekende technische indicatoren met de verwachte resultaten
        self.assertTrue(result['SMA_50'].equals(expected_sma_50))
        self.assertTrue(result['SMA_200'].equals(expected_sma_200))
        self.assertTrue(np.allclose(result['RSI'], expected_rsi, equal_nan=True))
        self.assertTrue(np.allclose(result['Upper_Band'], expected_upper_band, equal_nan=True))
        self.assertTrue(np.allclose(result['Lower_Band'], expected_lower_band, equal_nan=True))
        self.assertTrue(np.allclose(result['Stochastic_Oscillator'], expected_stochastic_oscillator, equal_nan=True))
        self.assertTrue(np.allclose(result['ATR'], expected_atr, equal_nan=True))
        self.assertTrue(np.allclose(result['EMA'], expected_ema, equal_nan=True))
        self.assertTrue(np.allclose(result['MACD'], expected_macd, equal_nan=True))
        self.assertTrue(np.allclose(result['MACD_Signal'], expected_macd_signal, equal_nan=True))
        self.assertTrue(np.allclose(result['MACD_Histogram'], expected_macd_histogram, equal_nan=True))
        self.assertTrue(np.allclose(result['Bollinger_Upper'], expected_bollinger_upper, equal_nan=True))
        self.assertTrue(np.allclose(result['Bollinger_Lower'], expected_bollinger_lower, equal_nan=True))


if __name__ == '__main__':
    unittest.main()

# Functie om de gegevens van de ai voor te bereiden
def prepare_data(data):
    data = data.dropna()
    X = data[['SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal_Line', 'EMA', 'Upper_Band', 'Lower_Band', 'Stochastic_Oscillator','ATR']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    return X_scaled, data['4. close'].shift(-1)

# functie om de voorbereide gegevens te testen
class TestPrepareData(unittest.TestCase):
    def test_prepare_data(self):
        # Voorbeeld invoerdata
        data = pd.DataFrame({
            'SMA_50': [10, 12, 15, 14, 16, np.nan, 17, 20, 22, 21],
            'SMA_200': [8, 11, 14, 15, np.nan, 13, 16, 18, 20, 19],
            'RSI': [np.nan, np.nan, 100.0, 88.8889, 100.0, 100.0, 85.1852, 100.0, 100.0, 97.9021],
            'MACD': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'Signal_Line': [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
            'EMA': [9.5, 11.1, 13.7, 14.9, 16.3, 17.9, 18.8, 19.7, 20.6, 21.4],  # voeg EMA-waarden toe
            'Upper_Band': [99, 101, 104, 103, 105, 107, 106, 109, 111, 110],  # voeg Upper_Band-waarden toe
            'Lower_Band': [97, 99, 102, 101, 103, 105, 104, 107, 109, 108],  # voeg Lower_Band-waarden toe
            'Stochastic_Oscillator': [20, 30, 40, 50, 60, 70, 80, 90, 95, 100],  # voeg Stochastic_Oscillator-waarden toe
            '4. close': [100, 102, 105, 104, 106, 108, 107, 110, 112, 111]
        })

        # Verwachte uitvoerdata
        expected_X_scaled = np.array([
            [0.33333333, 0.38461538, 1.        , 0.        , 0.        , 0.03030303, 0.5       , 0.33333333, 0.        ],
            [0.41666667, 0.46153846, 0.88888889, 0.11111111, 0.11111111, 0.13131313, 0.55555556, 0.44444444, 0.11111111],
            [0.5       , 0.53846154, 1.        , 0.22222222, 0.22222222, 0.23232323, 0.66666667, 0.55555556, 0.22222222],
            [0.46666667, 0.57692308, 0.88888889, 0.33333333, 0.33333333, 0.3030303 , 0.61111111, 0.5       , 0.33333333],
            [0.53333333, 0.61538462, 1.        , 0.44444444, 0.44444444, 0.39393939, 0.66666667, 0.55555556, 0.44444444],
            [0.56666667, 0.5       , 1.        , 0.55555556, 0.55555556, 0.53535354, 0.72222222, 0.61111111, 0.55555556],
            [0.6       , 0.65384615, 0.85185185, 0.66666667, 0.66666667, 0.58585859, 0.77777778, 0.66666667, 0.66666667],
            [0.66666667, 0.73076923, 1.        , 0.77777778, 0.77777778, 0.63636364, 0.88888889, 0.77777778, 0.77777778],
            [0.73333333, 0.80769231, 1.        , 0.88888889, 0.88888889, 0.68686869, 1.        , 0.88888889, 0.88888889]
        ])

        expected_y = pd.Series([102, 105, 104, 106, 108, 107, 110, 112, 111, np.nan])

        # Bereken voorbereide gegevens
        X_scaled, y = prepare_data(data)

        # Vergelijk de berekende voorbereide gegevens met de verwachte resultaten
        self.assertTrue(np.allclose(X_scaled, expected_X_scaled))
        self.assertTrue(y.equals(expected_y))

if __name__ == '__main__':
    unittest.main()

# Voorbeeld trainingsgegevens
X_train = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.2, 0.3, 0.4, 0.5, 0.6],
                    [0.3, 0.4, 0.5, 0.6, 0.7]])
y_train = np.array([1, 0, 1])

# Functie om het Keras-model te maken
def create_model(optimizer='adam', dropout_rate=0.0, kernel_regularizer=0.0):
    model = Sequential([
        Dense(units=50, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=regularizers.l2(kernel_regularizer)),
        Dropout(dropout_rate),
        Dense(units=30, activation='relu'),
        Dense(units=20, activation='relu'),
        Dense(units=10, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# KerasClassifier wrapper for Keras model
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define grid search parameters
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.2, 0.3, 0.4],
    'kernel_regularizer': [0.01, 0.02, 0.03]
}

# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)

# Print results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# vanaf hier beginnen we met het testtrainen van de ai 
class TestTrainModel(unittest.TestCase):
    def test_train_model(self):
        # Voorbeeld trainingsgegevens
        X_train = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                            [0.2, 0.3, 0.4, 0.5, 0.6],
                            [0.3, 0.4, 0.5, 0.6, 0.7]])
        y_train = np.array([1, 0, 1])

        # Train het model
        model = create_model()
        model.fit(X_train, y_train)  # Aanpassing hier

        # Voorspel de uitvoer met het getrainde model
        X_test = np.array([[0.4, 0.5, 0.6, 0.7, 0.8]])
        predictions = model.predict(X_test)

        # Controleer of de voorspellingen binnen het verwachte bereik liggen
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))

# Voer de unittests uit
if __name__ == '__main__':
    unittest.main()



# Functie om handelssignalen te genereren
def generate_signals(data, model):
    X, _ = prepare_data(data)
    signals = model.predict(X)
    return signals

# functie om de handelssignalen te testen
class TestGenerateSignals(unittest.TestCase):
    def test_generate_signals(self):
        # Voorbeeld data en model
        data = pd.DataFrame({
            'SMA_50': [0.1, 0.2, 0.3, 0.4, 0.5],
            'SMA_200': [0.2, 0.3, 0.4, 0.5, 0.6],
            'RSI': [0.3, 0.4, 0.5, 0.6, 0.7],
            'MACD': [0.4, 0.5, 0.6, 0.7, 0.8],
            'Signal_Line': [0.5, 0.6, 0.7, 0.8, 0.9],
            'EMA': [0.15, 0.25, 0.35, 0.45, 0.55],  # voeg EMA-waarden toe
            'Upper_Band': [0.2, 0.3, 0.4, 0.5, 0.6],  # voeg Upper_Band-waarden toe
            'Lower_Band': [0.05, 0.15, 0.25, 0.35, 0.45],  # voeg Lower_Band-waarden toe
            'Stochastic_Oscillator': [0.1, 0.2, 0.3, 0.4, 0.5],  # voeg Stochastic_Oscillator-waarden toe
            '4. close': [100, 102, 105, 104, 106]
        })

        model = Sequential([Dense(units=50, input_shape=(9,), activation='relu'),  # Let op: input_shape gewijzigd naar (9,)
                            Dense(units=1, activation='sigmoid')])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Genereer signalen
        signals = generate_signals(data, model)

        # Controleer of de signalen binnen het verwachte bereik liggen (tussen 0 en 1)
        self.assertTrue(np.all(signals >= 0))
        self.assertTrue(np.all(signals <= 1))

if __name__ == '__main__':
    unittest.main()




# Functie om handel uit te voeren op basis van signalen
def execute_trade(symbol, signals):
    last_signal = signals[-1]
    action = "Buying" if last_signal >= 0.5 else "Selling"
    print(f"{action} {symbol}.")

class TestExecuteTrade(unittest.TestCase):
    def test_execute_trade(self):
        # Voorbeeld signalen
        signals_buy = np.array([0.6, 0.7, 0.8, 0.9, 0.85])
        signals_sell = np.array([0.4, 0.3, 0.2, 0.1, 0.15])

        # Test "Buying" actie
        with patch('sys.stdout', new=StringIO()) as fake_out:
            execute_trade('AAPL', signals_buy)
            output = fake_out.getvalue().strip()
            self.assertEqual(output, 'Buying AAPL.')

        # Test "Selling" actie
        with patch('sys.stdout', new=StringIO()) as fake_out:
            execute_trade('AAPL', signals_sell)
            output = fake_out.getvalue().strip()
            self.assertEqual(output, 'Selling AAPL.')

if __name__ == '__main__':
    unittest.main()


# Hoofdfunctie om geautomatiseerd handelen uit te voeren
async def fetch_data_async(symbol, function, interval='5min', outputsize='full'):
    url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}"
    response = await asyncio.get_event_loop().run_in_executor(None, requests.get, url)
    if response.status_code == 200:
        data = response.json()
        if 'Time Series' in data:
            return symbol, pd.DataFrame(data['Time Series (5min)']).T
    print("Fout bij het ophalen van gegevens voor:", symbol)
    return symbol, None

# In auto_trade_async functie:
async def auto_trade_async(symbols):
    for symbol in symbols:
        # Haal de gegevens op
        symbol, data = await fetch_data_async(symbol, 'TIME_SERIES_INTRADAY')
        if data is not None:
            # Analyseer de gegevens
            analyzed_data = analyze_data(data)
            
            # Bereid de gegevens voor
            X, y = prepare_data(analyzed_data)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
            
            # Train het model
            model = create_model(X_train, y_train)
            
            # Genereer signalen
            signals = generate_signals(analyzed_data, model)
            
            # Voer handel uit op basis van signalen
            execute_trade(symbol, signals)

# Definieer de mock_data variabele met voorbeeldgegevens voor elk symbool
mock_data_stoxx50 = {
    'Time Series (5min)': {
        '2024-04-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-04-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        # Voeg meer voorbeelddata toe indien nodig
    }
}

mock_data_ndx = {
    # Mock data voor NDX symbool
}

mock_data_xauusd = {
    # Mock data voor XAUUSD symbool
}

mock_data_btcusd = {
    # Mock data voor BTCUSD symbool
}

mock_data_ethusd = {
    # Mock data voor ETHUSD symbool
}

mock_data_eurusd = {
    # Mock data voor EURUSD symbool
}

# Definieer de mock_fetch_data_async functie
async def mock_fetch_data_async(symbol, function, interval='5min', outputsize='full'):
    if symbol == 'STOXX50':
        return 'STOXX50', mock_data_stoxx50
    elif symbol == 'NDX':
        return 'NDX', mock_data_ndx
    elif symbol == 'XAUUSD':
        return 'XAUUSD', mock_data_xauusd
    elif symbol == 'BTCUSD':
        return 'BTCUSD', mock_data_btcusd
    elif symbol == 'ETHUSD':
        return 'ETHUSD', mock_data_ethusd
    elif symbol == 'EURUSD':
        return 'EURUSD', mock_data_eurusd
    else:
        return None, None

class TestAutoTradeAsync(unittest.IsolatedAsyncioTestCase):
    async def test_auto_trade_async(self):
        # Mock de functies die in auto_trade_async worden aangeroepen
        mock_calculate_technical_indicators = MagicMock()
        mock_prepare_data = MagicMock()
        mock_train_test_split = MagicMock()
        mock_train_model = MagicMock()
        mock_generate_signals = MagicMock()
        mock_execute_trade = MagicMock()

        # Pas de mocks toe in de functie
        with patch('auto_trade_async.fetch_data_async', mock_fetch_data_async), \
             patch('auto_trade_async.calculate_technical_indicators', mock_calculate_technical_indicators), \
             patch('auto_trade_async.prepare_data', mock_prepare_data), \
             patch('auto_trade_async.train_test_split', mock_train_test_split), \
             patch('auto_trade_async.train_model', mock_train_model), \
             patch('auto_trade_async.generate_signals', mock_generate_signals), \
             patch('auto_trade_async.execute_trade', mock_execute_trade):

            # Voer de auto_trade_async functie uit met een voorbeeldsymbool
            await auto_trade_async(['STOXX50'])

        # Controleer of de functies correct zijn aangeroepen
        mock_calculate_technical_indicators.assert_called_once()
        mock_prepare_data.assert_called_once()
        mock_train_test_split.assert_called_once()
        mock_train_model.assert_called_once()
        mock_generate_signals.assert_called_once()
        mock_execute_trade.assert_called_once()

if __name__ == '__main__':
    unittest.main()

# Start geautomatiseerd handelen voor elk opgegeven aandeel
symbols = ['STOXX50', 'NDX', 'XAUUSD', 'BTCUSD', 'ETHUSD', 'EURUSD']
iterations = 10
for _ in range(iterations):
    asyncio.run(auto_trade_async(symbols))
    time.sleep(86400)  # Wacht 1 dag tussen elke iteratie

# Na de lus voor geautomatiseerd handelen, voer de volgende stappen uit
# Voeg de TradingApp-klasse toe met demo-functionaliteit
class TradingApp:
    DEMO_BALANCE = 50000  # Nep balance voor demo-modus

    def __init__(self, demo_mode=False):
        self._demo_mode = demo_mode
        self._trading_balance = self.DEMO_BALANCE if demo_mode else 0

    @property
    def demo_mode(self):
        """Retourneert de huidige demo-modus status."""
        return self._demo_mode

    @property
    def trading_balance(self):
        """Retourneert het huidige handelsaldo."""
        return self._trading_balance

    def deposit_funds(self, amount):
        """Stort het opgegeven bedrag in de handelsrekening."""
        if self._demo_mode:
            print(f"Simulating deposit of {amount} in demo mode.")
            # Voer simulatieacties uit om geld toe te voegen aan de demo-balans
            self._trading_balance += amount
        else:
            print(f"Depositing {amount} in real mode.")
            # Voer echte acties uit om geld toe te voegen aan de handelsaccount

    def withdraw_funds(self, amount):
        """Neem het opgegeven bedrag op uit de handelsrekening."""
        if self._demo_mode:
            print(f"Simulating withdrawal of {amount} in demo mode.")
            # Voer simulatieacties uit om geld op te nemen van de demo-balans
            self._trading_balance -= amount
        else:
            print(f"Withdrawing {amount} in real mode.")
            # Voer echte acties uit om geld op te nemen van de handelsaccount

    def toggle_demo_mode(self):
        """Schakelt tussen demo-modus en echte modus."""
        self._demo_mode = not self._demo_mode
        if self._demo_mode:
            self._trading_balance = self.DEMO_BALANCE
        print("Demo mode is now", self._demo_mode)

# dit is een gui met pyside6
class TradingAppGUI:
    def __init__(self, master, trading_app):
        self.master = master
        self.trading_app = trading_app
        master.setWindowTitle("Trading App")

        # Frames voor verschillende secties van de GUI
        self.top_frame = QtWidgets.QFrame(master)
        self.top_frame.setLayout(QtWidgets.QHBoxLayout())
        self.middle_frame = QtWidgets.QFrame(master)
        self.middle_frame.setLayout(QtWidgets.QHBoxLayout())
        self.bottom_frame = QtWidgets.QFrame(master)
        self.bottom_frame.setLayout(QtWidgets.QHBoxLayout())

        master.layout = QtWidgets.QVBoxLayout()
        master.layout.addWidget(self.top_frame)
        master.layout.addWidget(self.middle_frame)
        master.layout.addWidget(self.bottom_frame)
        master.setLayout(master.layout)

        # Label en invoerveld voor het invoeren van het tickersymbool
        self.symbol_label = QtWidgets.QLabel("Enter Ticker Symbol:")
        self.symbol_entry = QtWidgets.QLineEdit()
        self.top_frame.layout.addWidget(self.symbol_label)
        self.top_frame.layout.addWidget(self.symbol_entry)

        # Knop om gegevens op te halen en grafiek te plotten
        self.plot_button = QtWidgets.QPushButton("Plot")
        self.plot_button.clicked.connect(self.plot_data)
        self.top_frame.layout.addWidget(self.plot_button)

        # Dropdownmenu voor het selecteren van de handelsstrategie
        self.strategy_label = QtWidgets.QLabel("Select Trading Strategy:")
        self.strategy_dropdown = QtWidgets.QComboBox()
        self.strategy_dropdown.addItems(['Simple Moving Average', 'RSI', 'MACD', 'EMA', 'Bollinger Bands', 'Stochastic Oscillator', 'ATR'])
        self.strategy_dropdown.setCurrentIndex(0)
        self.middle_frame.layout.addWidget(self.strategy_label)
        self.middle_frame.layout.addWidget(self.strategy_dropdown)

        # Canvas voor het plotten van de handelsgrafiek
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvas(self.fig)
        self.bottom_frame.layout.addWidget(self.canvas)

        # Initialisatie van variabelen voor balansinformatie
        self.trading_balance = 0
        self.available_balance = 0
        self.total_balance = 0

        # Update balansinformatie en plot
        self.update_balance()
        self.plot_balance()

        # Knoppen voor het storten en opnemen van geld
        self.deposit_button = QtWidgets.QPushButton("Deposit Funds")
        self.deposit_button.clicked.connect(self.deposit_funds)
        self.middle_frame.layout.addWidget(self.deposit_button)

        self.withdraw_button = QtWidgets.QPushButton("Withdraw Funds")
        self.withdraw_button.clicked.connect(self.withdraw_funds)
        self.middle_frame.layout.addWidget(self.withdraw_button)

        # Knop voor het toggelen van demo-modus
        self.demo_mode_button = QtWidgets.QPushButton("Toggle Demo Mode")
        self.demo_mode_button.clicked.connect(self.toggle_demo_mode)
        self.middle_frame.layout.addWidget(self.demo_mode_button)

    def update_balance(self):
        self.trading_balance = np.random.randint(1000, 5000)
        self.available_balance = np.random.randint(1000, 5000)
        self.total_balance = self.trading_balance + self.available_balance

    def plot_balance(self):
        plt.figure(figsize=(8, 6))
        plt.bar(['Trading Balance', 'Available Balance', 'Total Balance'], 
                [self.trading_balance, self.available_balance, self.total_balance], 
                color=['blue', 'green', 'red'])
        plt.title('Balance Information')
        plt.xlabel('Account Type')
        plt.ylabel('Balance')
        plt.show()

    def plot_data(self):
        # Haal gegevens op van Yahoo Finance
        symbol = self.symbol_entry.text()
        data = yf.download(symbol, start="2023-01-01", end="2024-01-01")

        # Voer technische analyse uit met behulp van StochAnalyzer
        stoch_values = self.stoch_analyzer.calculate_stochastic(data)

        # Plot de sluitingsprijs en Stochastic Oscillator-waarden
        self.ax.clear()
        self.ax.plot(data.index, data['Close'], label="Close Price")
        self.ax.plot(data.index, stoch_values, label="Stochastic Oscillator")
        self.ax.set_title(f"{symbol} Closing Price and Stochastic Oscillator")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price / Stochastic Oscillator")
        self.ax.legend()
        self.canvas.draw()

    def deposit_funds(self):
        # Hier kun je code toevoegen om het bedrag op te halen dat de gebruiker wil toevoegen
        amount = 100  # Dit is een voorbeeldbedrag, vervang dit door het daadwerkelijke bedrag dat de gebruiker invoert
        self.trading_app.deposit_funds(amount)

    def withdraw_funds(self):
        # Hier kun je code toevoegen om het bedrag op te halen dat de gebruiker wil opnemen
        amount = 100  # Dit is een voorbeeldbedrag, vervang dit door het daadwerkelijke bedrag dat de gebruiker invoert
        self.trading_app.withdraw_funds(amount)

    def toggle_demo_mode(self):
        self.trading_app.toggle_demo_mode()


# Functie om het hoofdvenster van de GUI te maken en uit te voeren
def run_trading_app():
    app = QtWidgets.QApplication([])
    root = QtWidgets.QMainWindow()
    trading_app = TradingApp()  # Maak een instantie van TradingApp
    trading_app_gui = TradingAppGUI(root, trading_app)  # Geef `root` en `trading_app` door aan TradingAppGUI
    root.show()
    app.exec()

# Voer de GUI-applicatie uit wanneer het script wordt uitgevoerd
if __name__ == "__main__":
    run_trading_app()