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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras import Dense
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

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

# Vul hier je eigen OANDA API-sleutel en account ID in
api_key = config.OANDA_API_KEY
account_id = config.OANDA_ACCOUNT_ID

# Maak een instantie van de ForexBrokerAPI-klasse
broker_api = ForexBrokerAPI(api_key, account_id)
       

# Vul hier je eigen API-sleutel in
api_key = config.api_key

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
def calculate_macd(close_price, window_short=12, window_long=26, window_signal=9):
    ema_short = close_price.ewm(span=window_short).mean()
    ema_long = close_price.ewm(span=window_long).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=window_signal).mean()
    return macd, signal_line

def calculate_rsi(close_price, window=14):
    delta = close_price.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_technical_indicators(data):
    data['SMA_50'] = data['4. close'].rolling(window=50).mean()
    data['SMA_200'] = data['4. close'].rolling(window=200).mean()
    data['RSI'] = calculate_rsi(data['4. close'])
    data['MACD'], data['Signal_Line'] = calculate_macd(data['4. close'])
    return data

# Functie om de gegevens voor het model voor te bereiden
def prepare_data(data):
    data = data.dropna()
    X = data[['SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal_Line']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    return X_scaled, data['4. close'].shift(-1)

# Functie om het AI-model te trainen
def train_model(X_train, y_train):
    model = Sequential([Dense(units=50, input_shape=(X_train.shape[1],), activation='relu'),
                        Dense(units=1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    return model

# Functie om handelssignalen te genereren
def generate_signals(data, model):
    X, _ = prepare_data(data)
    signals = model.predict(X)
    return signals

# Functie om handel uit te voeren op basis van signalen
def execute_trade(symbol, signals):
    last_signal = signals[-1]
    action = "Buying" if last_signal >= 0.5 else "Selling"
    print(f"{action} {symbol}.")

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

async def auto_trade_async(symbols):
    tasks = [fetch_data_async(symbol, 'TIME_SERIES_INTRADAY') for symbol in symbols]
    completed_tasks = await asyncio.gather(*tasks)
    for symbol, data in completed_tasks:
        if data is not None:
            data = calculate_technical_indicators(data)
            X, y = prepare_data(data)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    
            model = train_model(X_train, y_train)
            signals = generate_signals(data, model)
            execute_trade(symbol, signals)

# Start geautomatiseerd handelen voor elk opgegeven aandeel
symbols = ['STOXX50', 'NDX', 'XAUUSD', 'BTCUSD', 'ETHUSD', 'EURUSD']
iterations = 10
for _ in range(iterations):
    asyncio.run(auto_trade_async(symbols))
    time.sleep(86400)  # Wacht 1 dag tussen elke iteratie



# Na de lus voor geautomatiseerd handelen, voer de volgende stappen uit

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

# Creëer knoppen om het handelen te starten, te stoppen en geld terug te storten
start_button_ax = plt.axes([0.1, 0.05, 0.2, 0.075])
start_button = Button(start_button_ax, 'Start Trading')
start_button.on_clicked(start_trading)

stop_button_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
stop_button = Button(stop_button_ax, 'Stop Trading')
stop_button.on_clicked(stop_trading)

withdraw_button_ax = plt.axes([0.7, 0.05, 0.2, 0.075])
withdraw_button = Button(withdraw_button_ax, 'Withdraw Funds')
withdraw_button.on_clicked(lambda event: withdraw_funds())

# Functie om geld terug te storten van de app naar de gebruiker
def withdraw_funds():
    # Voer hier de logica uit om geld terug te storten naar de gebruiker
    # Dit kan het aanroepen van een externe API of het uitvoeren van een financiële transactie omvatten
    print("Geld wordt teruggestort naar de gebruiker.")

# Functie om de GUI bij te werken na het terugstorten van geld
def update_gui_after_withdrawal():
    # Voer hier de logica uit om de GUI bij te werken nadat geld is teruggestort
    # Bijvoorbeeld het bijwerken van balansinformatie of het weergeven van een melding aan de gebruiker
    print("GUI wordt bijgewerkt na het terugstorten van geld.")


# Initialisatie van variabelen voor balansinformatie
trading_balance = 0
available_balance = 0
total_balance = 0

# Update balansinformatie en plot
update_balance()
plot_balance()

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
