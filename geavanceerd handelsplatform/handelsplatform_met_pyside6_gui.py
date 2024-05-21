import numpy as np
import pandas as pd
import time
import requests
import csv
import json
import hashlib
import secrets
import sqlite3
import asyncio
import unittest
import yfinance as yf
import multiprocessing
import aiohttp
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
from sklearn.model_selection import GridSearchCV
from keras import KerasClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtWidgets import QAction, QMenuBar, QInputDialog, QMessageBox
from PyQt5.QtCore import QDateTime,Qt
from PySide6.QtGui import QIcon

# aanmaken van een gebruikers account 
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
    
# aanmaken van een database van de gebruikers , registreren van nieuwe gebruikers en het selecteren van de juiste gebruikers account
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

# autisiteren van de juiste gebruikers account
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
        
# aanroepen van de handelsbroker om toegang te verkrijgen tot de beurs in real-time        
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

# Functie om API-sleutels uit CSV-bestand op te halen
def get_api_keys_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        keys = next(reader)  # Veronderstelt dat de eerste rij in de CSV de sleutels bevat
        return keys

# CSV-bestandspad
csv_file_path = "D:/vscodedatasets/apisleutelhandel/api_franky/oanda_apisleutel_accound_id.csv"

# Haal de API-sleutels op uit het CSV-bestand
api_keys = get_api_keys_from_csv(csv_file_path)

# Haal de API-sleutels individueel op
oanda_api_key = api_keys.get('oanda_api_key', '')
oanda_account_id = api_keys.get('oanda_account_id', '')
mijn_eigen_api_sleutel = api_keys.get('mijn_eigen_api_sleutel', '')
api_key = api_keys.get('api_key', '')  
# Controleer of de API-sleutels zijn ingesteld
if oanda_api_key == "" or oanda_account_id == "" or mijn_eigen_api_sleutel == "":
    raise ValueError("Een of meer API-sleutels zijn niet ingesteld in het CSV-bestand")

# Maak een instantie van de ForexBrokerAPI-klasse
broker_api = ForexBrokerAPI(oanda_api_key, oanda_account_id)
payment_gateway=PaymentGateway(mijn_eigen_api_sleutel)

# de paralel functie verdeeld de taken die de ai moet maken over verschillende CPU(cores)
def parallel_task():
    # Voer parallelle taak uit
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

# functie om grafieken te maken met de technische indicators
def plot_technical_indicators(data, indicator_name, indicator_values):
    """
    Genereer een grafiek van een technische indicator.
    
    Parameters:
    - data (pandas.DataFrame): Pandas DataFrame met de invoerdata.
    - indicator_name (str): Naam van de technische indicator.
    - indicator_values (pandas.Series): Pandas Series met de waarden van de technische indicator.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    plt.plot(data.index, indicator_values, label=indicator_name, color='red')
    plt.title(f'{indicator_name} Analysis')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()  

# Functie om gegevens op te halen voor een specifiek aandeel    
def fetch_historical_data(symbol, function='TIME_SERIES_DAILY', outputsize='full'):
    """
    Fetch historical stock data for a specific symbol.
    
    Parameters:
    - symbol (str): The symbol of the stock for which data should be fetched.
    - function (str, optional): The type of data to fetch (default is 'TIME_SERIES_DAILY').
    - outputsize (str, optional): The size of the output (default is 'full').
    
    Returns:
    - pd.DataFrame or None: A DataFrame with the fetched historical data, or None if an error occurs.
    """
    # Check if symbol and function are strings
    if not isinstance(symbol, str) or not isinstance(function, str):
        raise TypeError("Symbol and function must be strings.")
    
    # Construct the URL for the API request
    url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize={outputsize}&apikey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        
        data = response.json()
        
        if 'Time Series (Daily)' in data:
            return pd.DataFrame(data['Time Series (Daily)']).T
        else:
            print("No valid data found for:", symbol)
            return None
            
    except requests.exceptions.RequestException as e:
        print("Error fetching data for:", symbol)
        print("Error message:", e)
        return None
    except json.JSONDecodeError as e:
        print("Error decoding JSON response for:", symbol)
        print("Error message:", e)
        return None

# functie om de juiste weg te vinden naar de juiste csv-bestanden 
def read_csv_file(file_path):
    """
    Read data from a CSV file.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing the CSV data.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError as e:
        print("File not found:", file_path)
        print("Error message:", e)
        return None
    except pd.errors.ParserError as e:
        print("Error parsing CSV file:", file_path)
        print("Error message:", e)
        return None
    
# Example usage
symbols = ['STOXX50', 'NDX', 'XAUUSD', 'BTCUSD', 'ETHUSD', 'EURUSD', 'N100', 'NVDA', 'TSLA']
combined_data = pd.DataFrame()

# Fetch historical data for each symbol and read corresponding CSV file
combined_data = pd.DataFrame()
for symbol in symbols:
    historical_data = fetch_historical_data(symbol)
    csv_file = f"D:/vscodedatasets/datasets/{symbol}_5yearhistory.csv"
    csv_data = read_csv_file(csv_file)

    if historical_data is not None and csv_data is not None:
        combined_data = pd.concat([combined_data, historical_data, csv_data], axis=0)
    else:
        print(f"Failed to fetch or read data for {symbol}")

""" vanaf hier beginnen de daadwerkelijke berekeningen om accurater te kunnen handelen en voorspellingen te doen aan de hand van real-time data en historische gegeven."""
# dit om de fibonacci berekening op te roepen
def generate_fibonacci(limit):
    """
    Generate Fibonacci numbers up to the specified limit.
    """
    fibonacci_sequence = [0, 1]
    while fibonacci_sequence[-1] + fibonacci_sequence[-2] <= limit:
        fibonacci_sequence.append(fibonacci_sequence[-1] + fibonacci_sequence[-2])
    return fibonacci_sequence

# Voorbeeld van het genereren van Fibonacci-getallen tot 100
fibonacci_levels = generate_fibonacci(100)              # door het getal op 100 te zetten geven we de ai een veel groter bereik om goed te kunnen vergelijken
print("Fibonacci-niveaus tot 100:", fibonacci_levels)

# functie om de  te berekenen
def calculate_atr(high_price, low_price, close_price, window=14):
    """
    Bereken de Gemiddelde Werkelijke Range (ATR) van de gegeven hoge, lage en slotprijzen met het opgegeven venster.
    
    Parameters:
    - high_price (pandas.Series): Pandas Series met de hoge prijzen.
    - low_price (pandas.Series): Pandas Series met de lage prijzen.
    - close_price (pandas.Series): Pandas Series met de slotprijzen.
    - window (int): Het venster voor de ATR-berekening. Standaard is 14.

    Returns:
    - atr (pandas.Series): Pandas Series met de ATR-waarden.
    """
    high_low = high_price - low_price
    high_close = np.abs(high_price - close_price.shift())
    low_close = np.abs(low_price - close_price.shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def is_fibonacci(number):
    """
    Controleren of het nummer een Fibonacci nummer is.
    """
    def is_perfect_square(x):
        return int(x ** 0.5) ** 2 == x

    return is_perfect_square(5 * number ** 2 + 4) or is_perfect_square(5 * number ** 2 - 4)

''' zo als je bij iedere berekkening zal zien zal er een testcase gemaakt worden ,
    deze om altijd zeker te zijn dat alle inkomende data goed en correct is .  '''

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

    def test_atr_fibonacci(self):
        # Voorbeeld invoerdata
        high_price = pd.Series([11, 13, 16, 15, 17, 19, 18, 21, 23, 22])
        low_price = pd.Series([9, 11, 14, 13, 15, 17, 16, 19, 21, 20])
        close_price = pd.Series([10, 12, 15, 14, 16, 18, 17, 20, 22, 21])

        # Bereken Average True Range (ATR)
        atr = calculate_atr(high_price, low_price, close_price)

        # Controleer of de ATR Fibonacci-niveaus bevat
        for value in atr:
            self.assertTrue(is_fibonacci(value))

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
    """
    Bereken de Relative Strength Index (RSI) van de gegeven sluitingsprijzen.

    Parameters:
    - close_price (pandas.Series): Een Pandas Series met sluitingsprijzen van aandelen.
    - window (int, optioneel): Het venster voor de RSI-berekening (standaard is 14).

    Returns:
    - rsi (pandas.Series): Een Pandas Series met RSI-waarden.
    """
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

# Functie om technische indicatoren te berekenen
def calculate_bollinger_bands(close_price, window=20, num_std=2):
    """
    Bereken de Bollinger Bands voor een reeks sluitingsprijzen.

    Parameters:
    - close_price (pandas.Series): Een Pandas Series met sluitingsprijzen van aandelen.
    - window (int, optioneel): Het venster voor het berekenen van het voortschrijdend gemiddelde (standaard is 20).
    - num_std (int, optioneel): Het aantal standaardafwijkingen voor het berekenen van de Bollinger Bands (standaard is 2).

    Returns:
    - upper_band (pandas.Series): Een Pandas Series met de bovengrens van de Bollinger Bands.
    - lower_band (pandas.Series): Een Pandas Series met de ondergrens van de Bollinger Bands.
    """
    rolling_mean = close_price.rolling(window=window).mean()
    rolling_std = close_price.rolling(window=window).std()
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std
    return upper_band, lower_band

# functie om technische indicators te berekenen
def calculate_stochastic_oscillator(close_price, high_price, low_price, window=14):
    """
    Bereken de Stochastische Oscillator voor een reeks sluitingsprijzen, hoogste prijzen en laagste prijzen.

    Parameters:
    - close_price (pandas.Series): Een Pandas Series met sluitingsprijzen van aandelen.
    - high_price (pandas.Series): Een Pandas Series met hoogste prijzen van aandelen.
    - low_price (pandas.Series): Een Pandas Series met laagste prijzen van aandelen.
    - window (int, optioneel): Het venster voor het berekenen van de Stochastische Oscillator (standaard is 14).

    Returns:
    - stochastic_oscillator (pandas.Series): Een Pandas Series met de waarden van de Stochastische Oscillator.
    """
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

# berekenen van technische indicators
def calculate_technical_indicators(data):
    data['SMA_50'] = data['4. close'].rolling(window=50).mean()
    data['SMA_200'] = data['4. close'].rolling(window=200).mean()
    data['RSI'] = calculate_rsi(data['4. close'])
    ema_window = 20  # bijvoorbeeld, kies het gewenste venster voor EMA
    data['EMA'] = calculate_ema(data['4. close'], ema_window)
    data['MACD'], data['Signal_Line'], _ = calculate_macd(data['4. close'])
    data['Upper_Band'], data['Lower_Band'] = calculate_bollinger_bands(data['4. close'])
    data['Stochastic_Oscillator'] = calculate_stochastic_oscillator(data['4. close'], data['2. high'], data['3. low'])
    data['ATR'] = calculate_atr(data['2. high'], data['3. low'], data['4. close'])
    
    return data

# testen van de technische indicators
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

# functie voor het analyseren en visualiseren van de indicatoren
def analyze_and_visualize_indicators(data):
    """
    Analyseer en visualiseer de technische indicatoren.
    
    Parameters:
    - data (pandas.DataFrame): Pandas DataFrame met de gegevens en berekende technische indicatoren.
    """
    # Plot de sluitingsprijs samen met SMA-lijnen
    plt.figure(figsize=(12, 8))
    plt.plot(data.index, data['4. close'], label='Close Price', color='blue')
    plt.plot(data.index, data['SMA_50'], label='SMA 50', color='orange')
    plt.plot(data.index, data['SMA_200'], label='SMA 200', color='green')
    plt.title('Closing Price with SMA Lines')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot RSI
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['RSI'], label='RSI', color='purple')
    plt.axhline(y=70, color='red', linestyle='--')
    plt.axhline(y=30, color='green', linestyle='--')
    plt.title('RSI Analysis')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot EMA
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['EMA'], label='EMA', color='magenta')
    plt.plot(data.index, data['4. close'], label='Close Price', color='blue', alpha=0.5)
    plt.title('EMA Analysis')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot MACD
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['MACD'], label='MACD', color='cyan')
    plt.plot(data.index, data['Signal_Line'], label='Signal Line', color='red', linestyle='--')
    plt.title('MACD Analysis')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot Bollinger Bands
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['4. close'], label='Close Price', color='blue')
    plt.plot(data.index, data['Upper_Band'], label='Upper Band', color='red')
    plt.plot(data.index, data['Lower_Band'], label='Lower Band', color='green')
    plt.title('Bollinger Bands Analysis')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot ATR
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['ATR'], label='ATR', color='orange')
    plt.title('ATR Analysis')
    plt.xlabel('Date')
    plt.ylabel('ATR')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot Stochastic Oscillator
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Stochastic_Oscillator'], label='Stochastic Oscillator', color='brown')
    plt.title('Stochastic Oscillator Analysis')
    plt.xlabel('Date')
    plt.ylabel('Stochastic Oscillator')
    plt.legend()
    plt.grid(True)
    plt.show()
# definieer en vul data met je gegevens
data=...
# Bereken technische indicatoren
data = calculate_technical_indicators(data)

# Analyseer en visualiseer technische indicatoren
analyze_and_visualize_indicators(data)
    
""" vanaf hier beginnen we al de verzamelde gegevens voor de ai voor te bereiden en testen zodanig deze goed getraind word. """
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
            'SMA_50': [10, 12, 15, 14, 16, np.nan, 17, 20, 22, 21],                                # voeg de korte bewegende waarde toe
            'SMA_200': [8, 11, 14, 15, np.nan, 13, 16, 18, 20, 19],                                # voeg de lange bewegende waarde toe
            'RSI': [np.nan, np.nan, 100.0, 88.8889, 100.0, 100.0, 85.1852, 100.0, 100.0, 97.9021], # voeg RSI-waarden toe
            'MACD': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],                            # voeg MACD-waarden toe
            'Signal_Line': [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],           # voeg SIGNAL_line waarden toe
            'EMA': [9.5, 11.1, 13.7, 14.9, 16.3, 17.9, 18.8, 19.7, 20.6, 21.4],                    # voeg EMA-waarden toe
            'Upper_Band': [99, 101, 104, 103, 105, 107, 106, 109, 111, 110],                       # voeg Upper_Band-waarden toe
            'Lower_Band': [97, 99, 102, 101, 103, 105, 104, 107, 109, 108],                        # voeg Lower_Band-waarden toe
            'Stochastic_Oscillator': [20, 30, 40, 50, 60, 70, 80, 90, 95, 100],                    # voeg Stochastic_Oscillator-waarden toe
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

# Definieer de handelsstrategie-klasse op laag, normaal, hoog risico
class TradingStrategy:
    def __init__(self, risk_level, profit_loss_ratio):
        self.risk_level = risk_level
        self.profit_loss_ratio = profit_loss_ratio

    def implement_strategy(self):
        # Criteria voor het bepalen van het risiconiveau op basis van winst-verliesratio
        if self.profit_loss_ratio >= 0.5:
            self.risk_level = 'low'
        elif 0.3 <= self.profit_loss_ratio < 0.5:
            self.risk_level = 'normal'
        else:
            self.risk_level = 'high'

        # Implementeer de handelsstrategie op basis van het risiconiveau
        if self.risk_level == 'low':
            # Voer handelsstrategie uit voor laag risico
            print("Uitvoeren van handelsstrategie voor laag risico.")
        elif self.risk_level == 'normal':
            # Voer handelsstrategie uit voor normaal risico
            print("Uitvoeren van handelsstrategie voor normaal risico.")
        elif self.risk_level == 'high':
            # Voer handelsstrategie uit voor hoog risico
            print("Uitvoeren van handelsstrategie voor hoog risico.")
        else:
            print("Ongeldig risiconiveau.")

# Voorbeeld van gebruik:
# Creëer een instantie van TradingStrategy met een risiconiveau en winst-verliesratio
strategy_low_risk = TradingStrategy(risk_level='low', profit_loss_ratio=0.6)
strategy_normal_risk = TradingStrategy(risk_level='normal', profit_loss_ratio=0.4)
strategy_high_risk = TradingStrategy(risk_level='high', profit_loss_ratio=0.2)

# Implementeer de strategie voor elk risiconiveau
strategy_low_risk.implement_strategy()
strategy_normal_risk.implement_strategy()
strategy_high_risk.implement_strategy()

# Voorbeeld trainingsgegevens x en y-train waarden
X_train = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.2, 0.3, 0.4, 0.5, 0.6],
                    [0.3, 0.4, 0.5, 0.6, 0.7]])
y_train = np.array([1, 0, 1])

# Voorbeeld testgegevens x en y-test waarden 
X_test = np.array([[0.4, 0.5, 0.6, 0.7, 0.8],
                   [0.5, 0.6, 0.7, 0.8, 0.9]])
y_test = np.array([0, 1])


# Functie om het Keras-model te maken met grid_search optimalisatie
def create_model(optimizer='adam', dropout_rate=0.0, kernel_regularizer=0.0):
    model = Sequential([
        Dense(units=50, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=regularizers.l2(kernel_regularizer)),
        Dropout(dropout_rate),
        Dense(units=40, activation='relu'),
        Dense(units=30, activation='relu'),
        Dense(units=25, activation='relu'),  # extra laag toegevoegd 25 neuronen
        Dense(units=20, activation='relu'),
        Dense(units=15, activation='relu'),  # Extra laag toegevoegd 15 neuronen
        Dense(units=10, activation='relu'),
        Dense(units=8, activation='relu'),   # Extra laag toegevoegd 8 neuronen
        Dense(units=6, activation='relu'),   # Extra laag toegevoegd 6 neuronen
        Dense(units=4, activation='relu'),   # Extra laag toegevoegd 4 neuronen
        Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# KerasClassifier wrapper for Keras model
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define grid search parameters
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # verschillende dropouts uitproberen           ''' de grid functie is er om het optimaliseren van de epochs, batch_size, dropout,verborgen lagen en leersnelheid ,
    'kernel_regularizer': [0.01, 0.02, 0.03, 0.04, 0.05], # verschillende leersnelheden uitproberen          dit wil zeggen da de ai op optimale prestaties kan presteren'''   
    'batch_size': [16, 32, 64, 128, 256],                 # verschillende batchgroottes uitproberen
    'epochs': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]   # verschillende aantallen epochs uitproberen
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

        # Voorbeeld testgegevens
        X_test = np.array([[0.4, 0.5, 0.6, 0.7, 0.8],
                           [0.5, 0.6, 0.7, 0.8, 0.9]])
        y_test = np.array([0, 1])

        # Train het model
        model = create_model()
        model.fit(X_train, y_train)

        # Voorspel de uitvoer met het getrainde model
        predictions = model.predict(X_test)

        # Controleer of de voorspellingen binnen het verwachte bereik liggen
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))

# Voer de unittests uit
if __name__ == '__main__':
    unittest.main()

def analyze_model_performance(model, X_test, y_test):
    """
    Analyseer de prestaties van het getrainde model op testgegevens.
    
    Parameters:
    - model: Het getrainde model.
    - X_test: Testkenmerken.
    - y_test: Ware labels voor de testgegevens.
    """
    # Evalueren van het model op testgegevens
    accuracy = model.evaluate(X_test, y_test)[1]
    print("Accuracy on test data:", accuracy)

    # Maak voorspellingen op testgegevens
    predictions = model.predict(X_test)

    # Genereer en toon de confusion matrix
    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(cm)

    # Traceer de ROC curve en bereken AUC
    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    auc_score = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Visualiseer voorspellingen versus ware labels
    plt.figure()
    plt.plot(y_test, label='True Labels')
    plt.plot(predictions, label='Predicted Labels')
    plt.title('True Labels vs Predicted Labels')
    plt.xlabel('Sample')
    plt.ylabel('Label')
    plt.legend()
    plt.show()

    # Optioneel: gebruik thresholds voor verdere analyse
    for i, threshold in enumerate(thresholds):
        print(f"Threshold: {threshold:.2f}, FPR: {fpr[i]:.2f}, TPR: {tpr[i]:.2f}")


# Functie om handelssignalen te genereren met threshold
def generate_signals(data, model, threshold=0.5):
    X, _ = prepare_data(data)
    probabilities = model.predict(X)  # Voorspel de waarschijnlijkheden
    signals = (probabilities > threshold).astype(int)  # Converteer de waarschijnlijkheden naar signalen op basis van de threshold
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

        # Genereer signalen met verschillende thresholds
        for threshold in [0.3, 0.5, 0.7]:
            signals = generate_signals(data, model, threshold=threshold)
            # Controleer of de signalen binnen het verwachte bereik liggen (tussen 0 en 1)
            self.assertTrue(np.all(signals >= 0))
            self.assertTrue(np.all(signals <= 1))

if __name__ == '__main__':
    unittest.main()

def analyze_signals(data, signals):
    # Voeg de gegenereerde signalen toe aan de dataset
    data['Signal'] = signals

    # Analyseer de signalen
    buy_signals = data[data['Signal'] >= 0.5]  # Signalen groter dan of gelijk aan 0.5 worden als "koop" geclassificeerd
    sell_signals = data[data['Signal'] < 0.5]  # Signalen kleiner dan 0.5 worden als "verkoop" geclassificeerd

    # Bereken de winst/verlies voor elke handelssignaal
    buy_profit = buy_signals['4. close'].diff().cumsum().iloc[-1]
    sell_profit = sell_signals['4. close'].diff().cumsum().iloc[-1]

    # Bereken de nauwkeurigheid van de signalen
    total_signals = len(data)
    correct_signals = len(data[data['Signal'] >= 0.5]) if buy_profit > 0 else len(data[data['Signal'] < 0.5])
    accuracy = correct_signals / total_signals * 100 if total_signals != 0 else 0

    # Print de resultaten
    print("Totaal aantal signalen:", total_signals)
    print("Aantal juiste signalen:", correct_signals)
    print("Nauwkeurigheid van de signalen:", accuracy)
    print("Winst bij verkoopsignalen:", sell_profit)
    print("Winst bij koopsignalen:", buy_profit)

    # Grafische analyse
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['4. close'], label='Prijs')
    plt.scatter(buy_signals.index, buy_signals['4. close'], marker='^', color='green', label='Koop Signalen')
    plt.scatter(sell_signals.index, sell_signals['4. close'], marker='v', color='red', label='Verkoop Signalen')
    plt.title('Handelssignalen')
    plt.xlabel('Datum')
    plt.ylabel('Prijs')
    plt.legend()
    plt.show()

# Genereer signalen
signals = generate_signals(data, model)

# Probeer verschillende drempelwaarden
for threshold in np.arange(0.3, 0.8, 0.1):  # Van 0.3 tot 0.7 met stapgrootte 0.1
    print(f"Analyse met drempel {threshold}:")
    analyze_signals(data, signals, threshold=threshold)



# Functie om handel uit te voeren op basis van signalen
def execute_trade(*symbols, signals, threshold=0.5):
    """
    Voer een handel uit op basis van de signalen.
    Parameters:
    - symbols: De symbolen van de activa waarop gehandeld wordt.
    - signals: De signalen op basis waarvan gehandeld wordt.
    - threshold: De drempelwaarde voor het bepalen van kopen of verkopen.
    """
    last_signal = signals[-1]
    action = "Buying" if last_signal >= threshold else "Selling"
    symbol_str = ','.join(symbols)
    print(f"{action} {symbol_str}.")

class TestExecuteTrade(unittest.TestCase):
    def test_execute_trade(self):
        # Voorbeeld signalen
        signals_buy = np.array([0.6, 0.7, 0.8, 0.9, 0.85])
        signals_sell = np.array([0.4, 0.3, 0.2, 0.1, 0.15])

        # Test "Buying" actie
        with patch('sys.stdout', new=StringIO()) as fake_out:
            execute_trade('STOXX50', 'NDX', 'BTCUSD', 'ETHUSD', 'EURUSD', 'XAUUSD', 'N100', 'NVDA', 'TSLA', signals=signals_buy, threshold=0.5)
            output = fake_out.getvalue().strip()
            self.assertEqual(output, 'Buying STOXX50,NDX,BTCUSD,ETHUSD,EURUSD,XAUUSD,N100,NVDA,TSLA.')

        # Test "Selling" actie
        with patch('sys.stdout', new=StringIO()) as fake_out:
            execute_trade('STOXX50', 'NDX', 'BTCUSD', 'ETHUSD', 'EURUSD', 'XAUUSD', 'N100', 'NVDA', 'TSLA', signals=signals_sell, threshold=0.5)
            output = fake_out.getvalue().strip()
            self.assertEqual(output, 'Selling STOXX50,NDX,BTCUSD,ETHUSD,EURUSD,XAUUSD,N100,NVDA,TSLA.')

if __name__ == '__main__':
    unittest.main()



def analyze_trades(symbols, signals, threshold=0.5):
    """
    Analyseer het aantal koop- en verkoopacties voor de gegeven symbolen en signalen.
    Parameters:
    - symbols: Lijst van symbolen van de activa.
    - signals: De signalen op basis waarvan gehandeld wordt.
    - threshold: De drempelwaarde voor het bepalen van kopen of verkopen.
    """
    # Tel het aantal koop- en verkoopacties
    buy_count = np.sum(signals >= threshold)
    sell_count = np.sum(signals < threshold)

    # Print de resultaten voor elk symbool
    symbol_str = ','.join(symbols)
    print(f"Aantal koopacties voor {symbol_str}: {buy_count}")
    print(f"Aantal verkoopacties voor {symbol_str}: {sell_count}")

# Voorbeeld gebruik
symbols = ['stoxx50', 'NDX', 'BTCUSD', 'ETHUSD', 'EURUSD', 'XAUUSD', 'N100', 'NVDA', 'TSLA']
signals = np.array([0.6, 0.7, 0.8, 0.9, 0.85, 0.4, 0.3, 0.2, 0.1, 0.15])

# Test met verschillende drempelwaarden
for threshold in np.arange(0.3, 0.8, 0.1):
    print(f"\nAnalyzing trades with threshold: {threshold}")
    analyze_trades(symbols, signals, threshold)

# Definieer de analyze_data functie om de gegevens te analyseren
def analyze_data(data):
    # Bereken het voortschrijdend gemiddelde (MA) over een bepaalde periode
    data['MA'] = data['close'].rolling(window=20).mean()  # 20-periode MA

    # Bereken de exponentieel voortschrijdend gemiddelde (EMA) over een bepaalde periode
    data['EMA'] = data['close'].ewm(span=12, adjust=False).mean()  # 12-periode EMA

    # Bereken de Average True Range (ATR) over een bepaalde periode
    data['TR'] = data['high'] - data['low']
    data['TR_mean'] = data['TR'].rolling(window=14).mean()  # 14-periode gemiddelde van TR
    data['ATR'] = data['TR_mean'].rolling(window=14).mean()  # 14-periode ATR

    # Bereken de Bollinger Bands
    data['SMA20'] = data['close'].rolling(window=20).mean()
    data['std_dev'] = data['close'].rolling(window=20).std()
    data['Upper_Band'] = data['SMA20'] + (data['std_dev'] * 2)
    data['Lower_Band'] = data['SMA20'] - (data['std_dev'] * 2)

    # Bereken de Stochastic Oscillator
    data['Lowest_Low'] = data['low'].rolling(window=14).min()
    data['Highest_High'] = data['high'].rolling(window=14).max()
    data['%K'] = (data['close'] - data['Lowest_Low']) / (data['Highest_High'] - data['Lowest_Low']) * 100
    data['%D'] = data['%K'].rolling(window=3).mean()

    # Bereken de MACD (12, 26, 9)
    data['EMA12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Bereken de Relative Strength Index (RSI)
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    data['RSI'] = 100 - (100 / (1 + RS))

    return data

def get_api_keys_from_csv(csv_file_path):
    # Functie om API-sleutels uit een CSV-bestand te halen
    df = pd.read_csv(csv_file_path)
    return df.set_index('key_name')['key_value'].to_dict()

async def fetch_data_async(symbol, function, interval=['1mon', '5MIN'], outputsize='full'):
    # Haal de API-sleutels op uit het CSV-bestand
    csv_file_path = "D:/vscodedatasets/apisleutelhandel/api_franky/oanda_apisleutel_accound_id.csv"
    api_keys = get_api_keys_from_csv(csv_file_path)

    # Haal de API-sleutel individueel op
    api_key = api_keys.get('oanda_api_key', '')

    if not api_key:
        raise ValueError("API-sleutel niet gevonden in het CSV-bestand")

    url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                if 'Time Series' in data:
                    # Controleren of de gewenste intervalgegevens beschikbaar zijn
                    interval_key = f'Time Series ({interval})'
                    if interval_key in data:
                        return symbol, pd.DataFrame(data[interval_key]).T
            print("Fout bij het ophalen van gegevens voor:", symbol)
            return symbol, None

async def fetch_all_data(symbols, functions):
    tasks = []
    for symbol in symbols:
        for function in functions:
            tasks.append(fetch_data_async(symbol, function))

    return await asyncio.gather(*tasks)

def load_csv_data(file_paths):
    csv_data = {}
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            csv_data[path] = df
        except Exception as e:
            print(f"Fout bij het laden van {path}: {e}")
    return csv_data

# Lijst van symbolen en functies
symbols = ['XAUUSD', 'STOXX50', 'BTCUSD', 'ETHUSD', 'EURUSD', 'N100', 'NDX', 'NVDA', 'TSLA']
functions = ['TIME_SERIES_MONTHLY', 'TIME_SERIES_DAY']

# Lijst van CSV-bestanden
csv_file_paths = [
    "D:/vscodedatasets/datasets/STOXX50_5yearhistory.csv",
    "D:/vscodedatasets/datasets/ndx-5yearhistory.csv",
    "D:/vscodedatasets/datasets/btcusd_5yearhistory.csv",
    "D:/vscodedatasets/datasets/ETHUSD_5yearhistory.csv",
    "D:/vscodedatasets/datasets/XAUUSD_5yearhistory.csv",
    "D:/vscodedatasets/datasets/N100-5yearhistory.csv",
    "D:/vscodedatasets/datasets/EURUSD_5yearhistory.csv",
    "D:/vscodedatasets/datasets/NVDA_5yearhistory.csv",
    "D:/vscodedatasets/datasets/TSLA_5yearhistory.csv"
]

# CSV-bestanden laden
csv_data = load_csv_data(csv_file_paths)

# Voorbeeld om gegevens weer te geven
for path, df in csv_data.items():
    print(f"Data uit {path}:\n", df.head())

# Voorbeeld van het asynchroon ophalen van API-gegevens
all_symbol_data = asyncio.run(fetch_all_data(symbols, functions))

# Verwerken en weergeven van opgehaalde gegevens
for data in all_symbol_data:
    symbol, df = data
    if df is not None:
        print(f"Data voor {symbol}:\n", df.head())

async def run_auto_trade(symbols):
    for symbol in symbols:
        await auto_trade_async(symbol)

async def analyze_data(data):
    # Voer hier je analyse uit op de gegeven data
    macd_line, signal_line, macd_histogram = calculate_macd(data)
    ema = calculate_ema(data)
    atr = calculate_atr(data)
    sma, upper_band, lower_band = calculate_bollinger_bands(data)
    k_percent, d_percent = calculate_stochastic_oscillator(data)
    rsi = calculate_rsi(data)

    # Voeg de berekende indicatoren toe aan de DataFrame
    data['MACD_line'] = macd_line
    data['MACD_signal_line'] = signal_line
    data['MACD_histogram'] = macd_histogram
    data['EMA'] = ema
    data['ATR'] = atr
    data['Bollinger_SMA'] = sma
    data['Bollinger_upper_band'] = upper_band
    data['Bollinger_lower_band'] = lower_band
    data['Stochastic_K_percent'] = k_percent
    data['Stochastic_D_percent'] = d_percent
    data['RSI]'] = rsi

    return data

# In auto_trade_async functie:
async def auto_trade_async(symbols, threshold=0.5):
    for symbol in symbols:
        symbol, data = await fetch_data_async(symbol, 'TIME_SERIES_INTRADAY', 'TIME_SERIES_INTRAMONTH')
        if data is not None:
            analyzed_data = await analyze_data(data)
            X, y = prepare_data(analyzed_data)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
            model = create_model(X_train, y_train)
            signals = generate_signals(analyzed_data, model, threshold)
            execute_trade(symbol, signals, threshold)

if __name__ == "__main__":
    import asyncio
    symbols = ['STOXX50', 'ETHUSD', 'BTCUSD', 'XAUUSD', 'NDX', 'EURUSD', 'N100', 'NVDA', 'TSLA']  # Lijst met symbolen om geautomatiseerde handel uit te voeren
    threshold = 0.5  # Stel de threshold in
    asyncio.run(auto_trade_async(symbols, threshold))
 
# Simuleer analyzed_data met voorbeeldgegevens
analyzed_data = {
    'Close': np.random.rand(100),  # Willekeurige voorbeeldgegevens voor 'Close' prijzen
    'EMA': np.random.rand(100),  # Willekeurige voorbeeldgegevens voor 'EMA'
    'MACD': np.random.rand(100),  # Willekeurige voorbeeldgegevens voor 'MACD'
    'ATR': np.random.rand(100),  # Willekeurige voorbeeldgegevens voor 'ATR'
    'Upper_Bollinger': np.random.rand(100),  # Willekeurige voorbeeldgegevens voor 'Upper_Bollinger'
    'Lower_Bollinger': np.random.rand(100),  # Willekeurige voorbeeldgegevens voor 'Lower_Bollinger'
    'Stoch': np.random.rand(100),  # Willekeurige voorbeeldgegevens voor 'Stoch'
    'RSI' : np.random.rand(100),  # willekeurige voorbeeldgegevens voor 'RSI' 
}

# Simuleer historische handelsgegevens met voorbeeldgegevens
data = {
    'Date': ['2024-04-26', '2024-04-27', '2024-03-26', '2024-02-26', '2023-04-26', 
             '2023-03-26', '2023-02-26', '2022-04-26', '2022-03-26', '2022-02-26'],
    'Open': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    'High': [110, 110, 110, 110, 110, 110, 110, 110, 110, 110],
    'Low': [90, 90, 90, 90, 90, 90, 90, 90, 90, 90],
    'Close': [105, 105, 105, 105, 105, 105, 105, 105, 105, 105],
    'Volume': [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]
}
df = pd.DataFrame(data) 
             
def bereken_totale_transactiekosten(df, transactiekosten):
    """
    Bereken de totale transactiekosten op basis van de transactiekosten en het aantal transacties.
    
    Parameters:
        - df: DataFrame: DataFrame met historische handelsgegevens.
        - transactiekosten: float: Kosten voor elke transactie als een percentage van de handelswaarde.
    
    Returns:
        - total_transaction_costs: float: De totale transactiekosten.
    """
    # Bereken het aantal transacties
    aantal_transacties = df['Position'].abs().sum()
    
    # Bereken de totale transactiekosten
    total_transaction_costs = aantal_transacties * transactiekosten
    
    return total_transaction_costs

def process_transaction_costs(total_transaction_costs, bank_account):
    """
    Verwerk de transactiekosten door deze af te trekken van het bankrekeningnummer.
    
    Parameters:
        - total_transaction_costs: float: De totale transactiekosten.
        - bank_account: str: Het bankrekeningnummer waarvan de transactiekosten moeten worden afgetrokken.
    """
    # Voeg code toe om de transactiekosten van het bankrekeningnummer af te trekken
    print(f"Transactiekosten van {total_transaction_costs} zijn afgetrokken van bankrekeningnummer {bank_account}")

# Definieer handelsparameters
# Vraag de gebruiker om het initiële kapitaal in te voeren
initial_capital = float(input("Voer het initiële kapitaal in: "))
transactiekosten = 0.005  # 0,5% transactiekosten
ma_venster = 20
bank_account= "jouw_kbc_rekeningnummer"

def run_auto_trading_strategy(df, initial_capital, transactiekosten, analyzed_data, bank_account):
    """
    Voer een eenvoudige handelsstrategie uit op basis van Moving Average (MA).
    
    Parameters:
        - df: DataFrame: DataFrame met historische handelsgegevens.
        - initial_capital: float: Het initiële kapitaal voor de handelsportefeuille.
        - transactiekosten: float: Kosten voor elke transactie als een percentage van de handelswaarde.
        - analyzed_data: DataFrame: DataFrame met geanalyseerde technische indicatoren.
        - bank_account: str: Het bankrekeningnummer waarop de transactiekosten moeten worden afgetrokken.
    
    Returns:
        - eindwaarde_portefeuille: float: De eindwaarde van de handelsportefeuille na het uitvoeren van de strategie.
        - totale_winst: float: De totale winst (of verlies) van de handelsstrategie.
    """
    # Hier voer je de handelsstrategie uit met behulp van de geanalyseerde data
    # Voeg technische indicatoren toe aan het DataFrame
    df = df.join(analyzed_data)
    # Voorbeeld van het toevoegen van meerdere indicatoren aan het DataFrame df
    df['MA'] = analyzed_data['EMA']  # Gebruik de berekende Exponential Moving Average (EMA) in plaats van een eenvoudige MA
    df['SMA'] = analyzed_data['SMA']  # Voeg een kolom toe voor Simple Moving Average (SMA)
    df['RSI'] = analyzed_data['RSI']  # Voeg een kolom toe voor Relative Strength Index (RSI)
    df['MACD'] = analyzed_data['MACD']  # Voeg een kolom toe voor Moving Average Convergence Divergence (MACD)
    df['ATR'] = analyzed_data['ATR']  # Voeg een kolom toe voor Average True Range (ATR)
    df['Upper_Bollinger'] = analyzed_data['Upper_Bollinger']  # Voeg een kolom toe voor de Bovenste Bollinger Band
    df['Lower_Bollinger'] = analyzed_data['Lower_Bollinger']  # Voeg een kolom toe voor de Onderste Bollinger Band
    df['Stoch'] = analyzed_data['Stoch']  # Voeg een kolom toe voor de Stochastic Oscillator

    # Simuleer transacties
    df['Position'] = df['Signal'].diff()
    
    # Bereken winst en verlies
    df['Winst/Verlies'] = df['Position'] * df['Close'] * (1 - transactiekosten)
    
    # Bereken eindwaarde en totale winst
    eindwaarde_portefeuille = initial_capital + df['Winst/Verlies'].sum()
    totale_winst = eindwaarde_portefeuille - initial_capital
    
    # Definieer de waarden voor total_transaction_costs en bank_account
    total_transaction_costs = bereken_totale_transactiekosten(df, transactiekosten)
    
    # Verwerk transactiekosten
    process_transaction_costs(total_transaction_costs, bank_account)
    
    return eindwaarde_portefeuille, totale_winst

# Voer een eenvoudige handelsstrategie uit op basis van het risiconiveau en de winst-verliesratio
def run_auto_trading_strategy_with_strategy(df, initial_capital, transactiekosten, analyzed_data, bank_account, trading_strategy):
    # Implementeer de strategie
    trading_strategy.implement_strategy()
    
    # Voer de handelsstrategie uit
    eindwaarde_portefeuille, totale_winst = run_auto_trading_strategy(df, initial_capital, transactiekosten, analyzed_data, bank_account)
    
    return eindwaarde_portefeuille, totale_winst

# Voorbeeld van gebruik:
# Creëer een instantie van TradingStrategy met een risiconiveau en winst-verliesratio
strategy_low_risk = TradingStrategy(risk_level='low', profit_loss_ratio=0.6)

# Voer de handelsstrategie uit met de gekozen strategie
eindwaarde, totale_winst = run_auto_trading_strategy_with_strategy(df, initial_capital, transactiekosten, analyzed_data, bank_account, strategy_low_risk)

# Stap 2: Roep de functie aan vanuit de app
eindwaarde_portefeuille, totale_winst = run_auto_trading_strategy(df, initial_capital, transactiekosten, analyzed_data, bank_account)

# Stap 3: Gebruik de geretourneerde waarden in de app
print("Eindwaarde van de portefeuille:", eindwaarde_portefeuille)
print("Totale winst:", totale_winst)


# Definieer de mock_data variabele met voorbeeldgegevens voor elk symbool
mock_data_stoxx50 = {
    'Time Series (5min)': {
        '2024-04-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-04-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-04-27 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-04-27 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-03-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-03-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-02-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-02-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2023-04-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2023-04-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2023-03-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2023-03-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
    }
}

mock_data_ndx = {
    'Time Series (5min)': {
        '2024-04-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-04-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-04-27 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-04-27 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-03-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-03-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-02-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-02-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2023-04-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2023-04-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2023-03-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2023-03-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
    }
}

mock_data_xauusd = {
    'Time Series (5min)': {
        '2024-04-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-04-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-04-27 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-04-27 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-03-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-03-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-02-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-02-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2023-04-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2023-04-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2023-03-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2023-03-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
    }
}

mock_data_btcusd = {
    'Time Series (5min)': {
        '2024-04-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-04-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-04-27 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-04-27 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-03-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-03-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-02-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-02-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2023-04-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2023-04-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2023-03-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2023-03-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
    }
}

mock_data_ethusd = {
    'Time Series (5min)': {
        '2024-04-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-04-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-04-27 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-04-27 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-03-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-03-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-02-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-02-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2023-04-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2023-04-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2023-03-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2023-03-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
    }
}

mock_data_eurusd = {
    'Time Series (5min)': {
        '2024-04-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-04-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-04-27 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-04-27 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-03-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-03-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-02-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-02-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2023-04-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2023-04-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2023-03-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2023-03-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
    }
}

mock_data_n100 = {
    'Time Series (5min)': {
        '2024-04-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-04-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-04-27 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-04-27 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-03-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-03-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-02-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-02-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2023-04-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2023-04-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2023-03-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2023-03-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
    }
}

mock_data_nvda = {
    'Time Series (5min)': {
        '2024-04-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-04-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-04-27 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-04-27 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-03-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-03-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-02-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-02-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2023-04-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2023-04-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2023-03-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2023-03-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
    }
}

mock_data_tsla = {
    'Time Series (5min)': {
        '2024-04-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-04-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-04-27 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-04-27 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-03-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-03-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2024-02-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2024-02-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2023-04-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2023-04-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
        '2023-03-26 09:30:00': {'open': '100', 'high': '110', 'low': '90', 'close': '105', 'volume': '10000'},
        '2023-03-26 09:35:00': {'open': '105', 'high': '115', 'low': '95', 'close': '100', 'volume': '15000'},
    }
}
# Definieer mock_data
mock_data = {
    'STOXX50': mock_data_stoxx50,
    'NDX': mock_data_ndx,
    'XAUUSD': mock_data_xauusd,
    'BTCUSD': mock_data_btcusd,
    'ETHUSD': mock_data_ethusd,
    'EURUSD': mock_data_eurusd,
    'N100' : mock_data_n100,
    'NVDA' : mock_data_nvda,
    'TSLA' : mock_data_tsla
}

# Definieer de mock_fetch_data_async functie
async def mock_fetch_data_async(symbol, function, interval='5min', outputsize='full'):
    if symbol in mock_data:
        return symbol, filter_mock_data(mock_data[symbol], function, interval, outputsize)
    return None, None

def filter_mock_data(data, function, interval, outputsize):
    filtered_data = {}
    for key, value in data.items():
        # Voeg hier logica toe om te filteren op basis van functie, interval en outputsize
        if function == 'TIME_SERIES_DAILY' and interval == 'daily' and outputsize == 'compact':
            # Voeg alleen dagelijkse gegevens toe met compacte outputgrootte
            filtered_data[key] = value
        elif function == 'TIME_SERIES_DAILY' and interval == 'daily' and outputsize == 'full':
            # Voeg alleen dagelijkse gegevens toe met volledige outputgrootte
            filtered_data[key] = value
        elif function == 'TIME_SERIES_INTRADAY' and interval == '5min' and outputsize == 'compact':
            # Voeg alleen gegevens van elk 5 minuten toe met compacte outputgrootte
            filtered_data[key] = value
        elif function == 'TIME_SERIES_INTRADAY' and interval == '5min' and outputsize == 'full':
            # Voeg alleen gegevens van elk 5 minuten toe met volledige outputgrootte
            filtered_data[key] = value
        # Voeg hier meer logica toe voor andere functie-interval-outputsize combinaties
    return filtered_data


class TestAutoTradeAsync(unittest.IsolatedAsyncioTestCase):
    async def test_auto_trade_async(self):
        # Mock de functies die in auto_trade_async worden aangeroepen
        mock_fetch_data_async = MagicMock()
        mock_calculate_technical_indicators = MagicMock()
        mock_prepare_data = MagicMock()
        mock_train_test_split = MagicMock()
        mock_train_model = MagicMock()
        mock_generate_signals = MagicMock()
        mock_execute_trade = MagicMock()

        # Stel de mock return values in indien nodig
        mock_fetch_data_async.return_value = ("symbol", {})  # Voeg een voorbeeld return value toe
        mock_calculate_technical_indicators.return_value = {}  # Voeg een voorbeeld return value toe
        mock_prepare_data.return_value = ([],[])  # Voeg een voorbeeld return value toe
        mock_train_test_split.return_value = ([], [], [], [])  # Voeg een voorbeeld return value toe
        mock_train_model.return_value = None  # Voeg een voorbeeld return value toe
        mock_generate_signals.return_value = []  # Voeg een voorbeeld return value toe

        # Pas de mocks toe in de functie
        with patch('auto_trade_async.fetch_data_async', mock_fetch_data_async), \
             patch('auto_trade_async.calculate_technical_indicators', mock_calculate_technical_indicators), \
             patch('auto_trade_async.prepare_data', mock_prepare_data), \
             patch('auto_trade_async.train_test_split', mock_train_test_split), \
             patch('auto_trade_async.train_model', mock_train_model), \
             patch('auto_trade_async.generate_signals', mock_generate_signals), \
             patch('auto_trade_async.execute_trade', mock_execute_trade):
            
            # Test met verschillende threshold waarden
            for threshold in [0.3, 0.5, 0.7]:
                with self.subTest(threshold=threshold):

                 # Voer de auto_trade_async functie uit met een voorbeeldsymbool en een threshold
                 await auto_trade_async(['STOXX50','NDX','ETHUSD','XAUUSD','EURUSD','BTCUSD','N100','NVDA','TSLA'], threshold=threshold)

                 # Controleer of de functies correct zijn aangeroepen
                 mock_fetch_data_async.assert_called_once()
                 mock_calculate_technical_indicators.assert_called_once()
                 mock_prepare_data.assert_called_once()
                 mock_train_test_split.assert_called_once()
                 mock_train_model.assert_called_once()
                 mock_generate_signals.assert_called_once()
                 mock_execute_trade.assert_called_once()

if __name__ == '__main__':
    unittest.main()

# Converteer de mock-gegevens naar dataframes voor alle stocks
dataframes = {}
for symbol, mock_data in mock_data.items():
    df = pd.DataFrame.from_dict(mock_data['Time Series (5min)'], orient='index')
    df.index = pd.to_datetime(df.index)
    dataframes[symbol] = df

# Voer de automatische handelsstrategie uit voor elke stock
eindwaarden = {}
totale_winsten = {}
for symbol, df in dataframes.items():
    eindwaarde_portefeuille, totale_winst = run_auto_trading_strategy(df, initial_capital, transactiekosten, ma_venster)
    eindwaarden[symbol] = eindwaarde_portefeuille
    totale_winsten[symbol] = totale_winst

# Toon de resultaten
for symbol in dataframes.keys():
    print(f"Voor {symbol}:")
    print("Eindwaarde portefeuille:", eindwaarden[symbol])
    print("Totale winst:", totale_winsten[symbol])

# vanaf hier word de balance,de winst , het verlies berekend en getest
def calculate_profit_and_loss(data):
    # Berekenen van P&L
    data['P&L'] = data['Aantal aandelen'] * (data['Verkoopprijs'] - data['Aankoopprijs'])
    total_PNL = data['P&L'].sum()
    return total_PNL

def test_calculate_profit_and_loss():
    # Maak een testdataset
    data = pd.DataFrame({
        'Aantal aandelen': [10, -5, 8],
        'Verkoopprijs': [20, 15, 25],
        'Aankoopprijs': [15, 10, 20]
    })

    # Voer de functie uit
    result = calculate_profit_and_loss(data)

    # Controleer het resultaat
    assert result == (10*(20-15) + (-5)*(15-10) + 8*(25-20)), "Total P&L is not calculated correctly"

def calculate_sharpe_ratio(data):
    # Berekenen van gemiddeld rendement en standaarddeviatie
    average_return = data['Rendement'].mean()
    volatility = data['Rendement'].std()

    # Risicovrije rente (bijv. 0.02 voor 2%)
    risk_free_rate = 0.02

    # Berekenen van Sharpe ratio
    sharpe_ratio = (average_return - risk_free_rate) / volatility
    return sharpe_ratio

def test_calculate_sharpe_ratio():
    # Maak een testdataset
    data = pd.DataFrame({
        'Rendement': [0.02, 0.03, -0.01, 0.05, 0.01]
    })

    # Voer de functie uit
    result = calculate_sharpe_ratio(data)

    # Controleer het resultaat
    assert abs(result - ((0.03-0.02)/0.02)) < 1e-5, "Sharpe ratio is not calculated correctly"

def calculate_max_drawdown(data):
    # Berekenen van maximale drawdown
    peak = data['Portefeuillewaarde'].cummax()
    drawdown = (data['Portefeuillewaarde'] - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown

def test_calculate_max_drawdown():
    # Maak een testdataset
    data = pd.DataFrame({
        'Portefeuillewaarde': [100, 120, 90, 110, 80]
    })

    # Voer de functie uit
    result = calculate_max_drawdown(data)

    # Controleer het resultaat
    assert result == -0.2, "Max drawdown is not calculated correctly"

def calculate_win_percentage(data):
    # Berekenen van winstpercentage
    winning_trades = data[data['P&L'] > 0]
    win_percentage = (len(winning_trades) / len(data)) * 100
    return win_percentage

def test_calculate_win_percentage():
    # Maak een testdataset
    data = pd.DataFrame({
        'P&L': [100, -50, 75, -30, 120, -20, 60, -10]
    })

    # Voer de functie uit
    result = calculate_win_percentage(data)

    # Controleer het resultaat
    assert abs(result - (5/8)*100) < 1e-5, "Win percentage is not calculated correctly"

def calculate_average_profit_per_trade(data):
    # Berekenen van gemiddelde winst per winnende transactie
    winning_trades = data[data['P&L'] > 0]
    average_profit_per_trade = winning_trades['P&L'].mean()
    return average_profit_per_trade

# Voorbeeld van gebruik:
# result_PNL = calculate_profit_and_loss(data)
# result_sharpe = calculate_sharpe_ratio(data)
# result_max_drawdown = calculate_max_drawdown(data)
# result_win_percentage = calculate_win_percentage(data)
# result_average_profit_per_trade = calculate_average_profit_per_trade(data)
# Start geautomatiseerd handelen voor elk opgegeven aandeel
symbols = ['STOXX50', 'NDX', 'XAUUSD', 'BTCUSD', 'ETHUSD', 'EURUSD','N100','NVDA','TSLA']
iterations = 10
loop = asyncio.get_event_loop()
for _ in range(iterations):
    loop.run_until_complete(run_auto_trade(symbols))
    time.sleep(86400)  # Wacht 1 dag tussen elke iteratie
loop.close()

# Voeg de TradingApp-klasse toe met demo-functionaliteit
class TradingApp:
    DEMO_BALANCE = 50000  # Nep balance voor demo-modus

    def __init__(self, demo_mode=False):
        self._demo_mode = demo_mode
        self._trading_balance = self.DEMO_BALANCE if demo_mode else 0
        self._risk_level = "normaal"  # Standaard risiconiveau

    @property
    def risk_level(self):
        """Retourneert het huidige risiconiveau."""
        return self._risk_level

    def set_risk_level(self, risk_level):
        """Stelt het risiconiveau in op het opgegeven niveau."""
        self._risk_level = risk_level
        print("Risiconiveau ingesteld op:", risk_level)

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

    def start_autotrading(self):
     self.trading_app.start_autotrading()

    def stop_autotrading(self):
     self.trading_app.stop_autotrading()
    




# dit is een gui met pyside6
class TradingAppGUI:
    def __init__(self, master, trading_app):
        self.master = master
        self.trading_app = trading_app
        master.setWindowTitle("Trading App")
        self.menu_bar= None
        
        # Voeg knoppen toe voor het selecteren van risiconiveaus
        self.low_risk_button = QtWidgets.QPushButton("laag Risico")
        self.low_risk_button.clicked.connect(self.set_low_risk_level)
        self.middle_frame.layout.addWidget(self.low_risk_button)

        self.normal_risk_button = QtWidgets.QPushButton("Normaal Risico")
        self.normal_risk_button.clicked.connect(self.set_normal_risk_level)
        self.middle_frame.layout.addWidget(self.normal_risk_button)

        self.high_risk_button = QtWidgets.QPushButton("Hoog Risico")
        self.high_risk_button.clicked.connect(self.set_high_risk_level)
        self.middle_frame.layout.addWidget(self.high_risk_button)

        # Frames voor verschillende secties van de GUI
        self.top_frame.setStyleSheet("background-color: lightgray; border: 1px solid gray;") # dit is een lichtgrijzen achtergrond
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

        # Voeg labels toe om feedback aan de gebruiker te geven
        self.symbol_label.setStyleSheet("font-weight: bold")  # Vetgedrukte tekst voor het tickersymbool label
        self.status_label.setStyleSheet("color: green")  # Groene tekst voor succesvolle acties
        self.status_label = QtWidgets.QLabel("")
        self.middle_frame.layout.addWidget(self.status_label)

        # Voeg sneltoetsen toe voor veelgebruikte functies
        self.deposit_button.setStyleSheet("background-color: #4CAF50; color: white; border: none; padding: 10px 24px;") # dit is een groene achtergrond
        self.deposit_action = QAction("Storten", self.master)
        self.deposit_action.triggered.connect(self.deposit_funds)
        self.deposit_action.setShortcut("Ctrl+S")
        self.withdraw_button.setStyleSheet("backgrond-color: #2196f3; color: white; border: none; padding: 10px 24px;") # dit is een blauwe achtergrond
        self.withdraw_action = QAction("Opnemen", self.master)
        self.withdraw_action.triggered.connect(self.withdraw_funds)
        self.withdraw_action.setShortcut("Ctrl+W")
        self.demo_mode_button.setStyleSheet("backgrond-color: #ff9800; color: white; border: none; padding: 10px 24px;") # dit is een orange achtergrond
        self.demo_mode_action = QAction("Demo-modus togglen", self.master)
        self.demo_mode_action.triggered.connect(self.toggle_demo_mode)
        self.demo_mode_action.setShortcut("Ctrl+D")

        # Voeg een actie toe voor het helpmenu
        self.help_button.setStyleSheet("background-color: #FFFF00; color: white; border: none; padding: 10px 24px;") # dit is een gele achtergrond
        self.help_action = QAction("Help", self.master)
        self.help_action.triggered.connect(self.show_help)

        # Voeg het helpmenu toe aan de menubalk
        menu_bar = QMenuBar(self.master)
        self.master.setMenuBar(menu_bar)
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction(self.help_action)

        # Voeg een lijstwidget toe voor de transactiegeschiedenis
        self.transaction_history_list.setStyleSheet("background-color: lightblue;") # dit iseen licht blauwe achtergrond
        self.transaction_history_list = QtWidgets.QListWidget()
        self.bottom_frame.layout.addWidget(self.transaction_history_list)
        
        # knoppen voor starten en stoppen van de autotrade functie
        self.start_autotrading_button = QtWidgets.QPushButton("Start Autotrading")
        self.start_autotrading_button.setStyleSheet("background-color: #4CAF50; color: white; border: none; padding: 10px 24px;")
        self.start_autotrading_button.clicked.connect(self.start_autotrading)
        self.middle_frame.layout.addWidget(self.start_autotrading_button)

        self.stop_autotrading_button = QtWidgets.QPushButton("Stop Autotrading")
        self.stop_autotrading_button.setStyleSheet("background-color: #f44336; color: white; border: none; padding: 10px 24px;")
        self.stop_autotrading_button.clicked.connect(self.stop_autotrading)
        self.middle_frame.layout.addWidget(self.stop_autotrading_button)

        # Knoppen voor het storten en opnemen van geld
        self.deposit_button = QtWidgets.QPushButton("Deposit Funds")
        self.deposit_button.setIcon(QIcon("deposit_icon.png"))
        self.deposit_button.clicked.connect(self.deposit_funds)
        self.middle_frame.layout.addWidget(self.deposit_button)

        self.withdraw_button = QtWidgets.QPushButton("Withdraw Funds")
        self.withdraw_button.clicked.connect(self.withdraw_funds)
        self.middle_frame.layout.addWidget(self.withdraw_button)

        # Knop voor het toggelen van demo-modus
        self.demo_mode_button = QtWidgets.QPushButton("Toggle Demo Mode")
        self.demo_mode_button.clicked.connect(self.toggle_demo_mode)
        self.middle_frame.layout.addWidget(self.demo_mode_button)

        # Knop om gegevens op te halen en grafiek te plotten
        self.plot_button = QtWidgets.QPushButton("Plot")
        self.plot_button.clicked.connect(self.plot_data)
        self.top_frame.layout.addWidget(self.plot_button)

        # Label en invoerveld voor het invoeren van het tickersymbool
        self.symbol_label = QtWidgets.QLabel("Enter Ticker Symbol:")
        self.symbol_entry = QtWidgets.QLineEdit()
        self.top_frame.layout.addWidget(self.symbol_label)
        self.top_frame.layout.addWidget(self.symbol_entry)

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

        # Stel labels in voor balansinformatie
        self.trading_balance_label = QtWidgets.QLabel("Trading Balance: " + str(self.trading_balance))
        self.available_balance_label = QtWidgets.QLabel("Available Balance: " + str(self.available_balance))
        self.total_balance_label = QtWidgets.QLabel("Total Balance: " + str(self.total_balance))

        # Stel labels in voor balansinformatie
        self.trading_balance_label = QtWidgets.QLabel("Trading Balance: " + str(self.trading_balance))
        self.available_balance_label = QtWidgets.QLabel("Available Balance: " + str(self.available_balance))
        self.total_balance_label = QtWidgets.QLabel("Total Balance: " + str(self.total_balance))

        # Pas de achtergrondkleur van de labels aan
        self.trading_balance_label.setStyleSheet("background-color: lightblue;")
        self.available_balance_label.setStyleSheet("background-color: lightgreen;")
        self.total_balance_label.setStyleSheet("background-color: lightyellow;")

        # Voeg labels toe aan het bovenste frame
        self.top_frame.layout.addWidget(self.trading_balance_label)
        self.top_frame.layout.addWidget(self.available_balance_label)
        self.top_frame.layout.addWidget(self.total_balance_label)

        # Pas de achtergrondkleur van de labels aan op basis van de balanswaarden
        if self.trading_balance < 0:
            self.trading_balance_label.setStyleSheet("background-color: red;")
        else:
            self.trading_balance_label.setStyleSheet("background-color: green;")

        if self.available_balance < 0:
            self.available_balance_label.setStyleSheet("background-color: red;")
        else:
           self.available_balance_label.setStyleSheet("background-color: green;")

        if self.total_balance < 0:
           self.total_balance_label.setStyleSheet("background-color: red;")
        else:
           self.total_balance_label.setStyleSheet("background-color: green;")

        # Update balansinformatie en plot
        self.update_balance()
        self.plot_balance()

    def set_low_risk_level(self):
        """stelt het risiconiveau in op laag."""
        self.trading_app.set_risk_level("laag")

    def set_normal_risk_level(self):
        """Stelt het risiconiveau in op normaal."""
        self.trading_app.set_risk_level("normaal")

    def set_high_risk_level(self):
        """Stelt het risiconiveau in op hoog."""
        self.trading_app.set_risk_level("hoog")

    def update_transaction_history(self, transaction):
        """Voegt een nieuwe transactie toe aan de transactiegeschiedenis."""
        transaction_time = QDateTime.currentDateTime().toString(Qt.DefaultLocaleLongDate)
        status = "Succesvol" if transaction["success"] else "Mislukt"
        transaction_details = f"{transaction_time} - Bedrag: {transaction['amount']}, Status: {status}"
        self.transaction_history_list.addItem(transaction_details)
            
    def show_help(self):
        # Toon een eenvoudig helpbericht aan de gebruiker
        QtWidgets.QMessageBox.information(self.master, "Help", "Welkom bij de Trading App!\n\nDit programma stelt gebruikers in staat om handelsgegevens te bekijken, stortingen en opnames te doen, en te schakelen tussen demo- en echte modus.")
        self.menu_bar= QMenuBar(self.master)  
        self.master.setMenubar(self.menu_bar)  
        # Voeg een menu toe aan de GUI
        file_menu = self.menu_bar.addMenu("Bestand")  # Gebruik self.menu_bar
        file_menu.addAction(self.deposit_action)
        file_menu.addAction(self.withdraw_action)

        options_menu = self.menu_bar.addMenu("Opties")  # Gebruik self.menu_bar
        options_menu.addAction(self.demo_mode_action)

    def update_balance(self):
        # Veronderstel dat self.trading_app een instantie is van je handelsapplicatie
        self.trading_balance = self.trading_app.get_trading_balance()
        self.available_balance = self.trading_app.get_available_balance()
        self.total_balance = self.trading_app.get_total_balance()

        # Werk de labeltekst bij met de nieuwe balansinformatie
        self.trading_balance_label.setText("Trading Balance: " + str(self.trading_balance))
        self.available_balance_label.setText("Available Balance: " + str(self.available_balance))
        self.total_balance_label.setText("Total Balance: " + str(self.total_balance))

    def plot_balance(self):
        plt.figure(figsize=(8, 6))
        plt.bar(['Trading Balance', 'Available Balance', 'Total Balance'], 
            [self.trading_balance, self.available_balance, self.total_balance], 
            color=['lightblue', 'lightgreen', 'lightyellow'])  # Gebruik vergelijkbare kleuren
        plt.title('Balance Information')
        plt.xlabel('Account Type')
        plt.ylabel('Balance')
        plt.show()

    def plot_data(self):
        # Haal gegevens op van Yahoo Finance
        symbol = self.symbol_entry.text()
        if not symbol:
            QtWidgets.QMessageBox.warning(self.master, "Waarschuwing", "Voer alstublieft een ticker-symbool in.")
            return
        data = yf.download(symbol, start="2023-01-01", end="2024-01-01")

        # Voer technische analyse uit met behulp van verschillende indicatoren
        stoch_values = self.stoch_analyzer.calculate_stochastic(data)
        atr_values = calculate_atr(data['High'], data['Low'], data['Close'])
        macd_values = calculate_macd(data['Close'])
        ema_values = calculate_ema(data['Close'])
        bollinger_bands_values = calculate_bollinger_bands(data['Close'])
        rsi_values = calculate_rsi(data['Close'])

        # Plot de sluitingsprijs en verschillende technische indicatoren
        self.ax.clear()
        self.ax.plot(data.index, data['Close'], label="Close Price")
        self.ax.plot(data.index, stoch_values, label="Stochastic Oscillator")
        self.ax.plot(data.index, atr_values, label="ATR")
        self.ax.plot(data.index, macd_values, label="MACD")
        self.ax.plot(data.index, ema_values, label="EMA")
        self.ax.plot(data.index, bollinger_bands_values[0], label="Bollinger Bands Upper")
        self.ax.plot(data.index, bollinger_bands_values[1], label="Bollinger Bands Lower")
        self.ax.plot(data.index, rsi_values, label="RSI")
        self.ax.set_title(f"{symbol} Closing Price and Technical Indicators")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price / Indicator Value")
        self.ax.legend()
        self.canvas.draw()

    def is_valid_amount(amount):
        try:
            amount = float(amount)
            if amount <= 0:
             return False
            return True
        except ValueError:
           return False
        
# Functie om een bevestigingsdialoogvenster weer te geven
def show_confirmation_dialog(message, parent=None):
    reply = QMessageBox.question(parent, 'Bevestiging', message, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    return reply == QMessageBox.Yes
result = show_confirmation_dialog("wil u doorgaan?")
   
# Functie om geld te storten
def deposit_funds(self):
    # Toon een dialoogvenster om de gebruiker het bedrag te laten invoeren
    amount, ok = QInputDialog.getDouble(self.master, "Storten", "Voer het bedrag in:")
    if ok:
        # Bevestig of de gebruiker de actie wil uitvoeren
        confirmation_message = f"Weet je zeker dat je {amount} wilt storten?"
        if show_confirmation_dialog(confirmation_message):
            # Voer stortingsactie uit
            self.trading_app.deposit_funds(amount)
            # Geef feedback aan de gebruiker
            if self.trading_app.demo_mode:
                self.status_label.setText(f"Storting van {amount} in demo-modus succesvol uitgevoerd.")
                self.status_label.setStyleSheet("color: green")  # Groene tekst voor succesvolle acties
            else:
                self.status_label.setText(f"Storting van {amount} in real mode succesvol uitgevoerd.")
                self.status_label.setStyleSheet("color: green")
        else:
            self.status_label.setText("Storten geannuleerd.")
            self.status_label.setStyleSheet("color: red")  # Rode tekst voor geannuleerde acties


def withdraw_funds(self):
    if self.trading_app.demo_mode:
        # Geef een melding dat opnames niet zijn toegestaan in de demo-modus
        self.status_label.setText("Opnames zijn niet toegestaan in de demo-modus.")
        self.status_label.setStyleSheet("color: red")  # Rode tekst voor foutmeldingen
    else:
        # Toon een dialoogvenster om de gebruiker het bedrag te laten invoeren
        amount, ok = QInputDialog.getDouble(self.master, "Opnemen", "Voer het bedrag in:")
        if ok:
            # Voer opnameactie uit
            self.trading_app.withdraw_funds(amount)
            # Geef feedback aan de gebruiker
            self.status_label.setText(f"Opname van {amount} in real mode succesvol uitgevoerd.")
            self.status_label.setStyleSheet("color: green")  # Groene tekst voor succesvolle acties
        else:
            # Gebruiker heeft geannuleerd, geef feedback
            self.status_label.setText("Opnemen geannuleerd.")
            self.status_label.setStyleSheet("color: red")  # Rode tekst voor geannuleerde acties

def toggle_demo_mode(self):
        # Voer actie uit om demo-modus te wijzigen
        self.trading_app.toggle_demo_mode()
        # Geef feedback aan de gebruiker
        mode = "ingeschakeld" if self.trading_app.demo_mode else "uitgeschakeld"
        self.status_label.setText(f"Demo-modus is nu {mode}.")


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