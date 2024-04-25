import numpy as np
import pandas as pd
import time
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras import Dense
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Definieer de PaymentGateway klasse voor betalingsverwerking
class PaymentGateway:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://paymentgatewayapi.com/api"

    def process_payment(self, amount, currency, user_id):
        # Stuur een verzoek naar de betalingsgateway API om een betaling te verwerken
        endpoint = f"{self.base_url}/process_payment"
        payload = {
            "api_key": self.api_key,
            "amount": amount,
            "currency": currency,
            "user_id": user_id
        }
        response = requests.post(endpoint, json=payload)
        
        # Controleer of het verzoek succesvol was en verwerk de respons
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return True, data.get("transaction_id")
            else:
                return False, data.get("error_message")
        else:
            return False, "Fout bij het verwerken van de betaling"

# Vul hier je eigen API-sleutel in
api_key = 'YOUR_API_KEY'

# Functie om gegevens op te halen voor een specifiek aandeel
def fetch_data(symbol, interval='5min', outputsize='full'):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}",
    "https://www.alphavantage.co/query?function=VWAP&symbol=STOXX50,NDX,XAUUSD,BTCUSD,ETHUSD,EURUSD&interval=15min&apikey=" + api_key,
    "https://www.alphavantage.co/query?function=MACDEXT&symbol=STOXX50,NDX,XAUUSD,BTCUSD,ETHUSD,EURUSD&interval=daily&series_type=open&apikey=" + api_key,
    "https://www.alphavantage.co/query?function=MACD&symbol=STOXX50,NDX,XAUUSD,BTCUSD,ETHUSD,EURUSD&interval=daily&series_type=open&apikey=" + api_key,
    "https://alphavantageapi.co/timeseries/running_analytics?SYMBOLS=STOXX50,NDX,XAUUSD,BTCUSD,ETHUSD,EURUSD&RANGE=2month&INTERVAL=DAILY&OHLC=Close&WINDOW_SIZE=20&CALCULATIONS=MEAN,STDDEV(annualized=True)&apikey=" +api_key,
    "https://www.alphavantage.co/query?function=OVERVIEW&symbol=STOXX50,NDX,XAUUSD,BTCUSD,ETHUSD,EURUSD&apikey=" + api_key,
    "https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol=STOXX50,NDX,XAUUSD,BTCUSD,ETHUSD,EURUSD&apikey=" + api_key,
    "https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol=STOXX50,NDX,XAUUSD,BTCUSD,ETHUSD,EURUSD&apikey=" + api_key,
    "https://www.alphavantage.co/query?function=CASH_FLOW&symbol=STOXX50,NDX,XAUUSD,BTCUSD,ETHUSD,EURUSD&apikey=" + api_key,
    "https://www.alphavantage.co/query?function=EARNINGS&symbol=STOXX50,NDX,XAUUSD,BTCUSD,ETHUSD,EURUSD&apikey=" + api_key,
    "https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&symbol=STOXX50,NDX,XAUUSD,BTCUSD,ETHUSD,EURUSD&horizon=12month&apikey=" + api_key,
    "https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol=EUR&to_symbol=USD&interval=5min&outputsize=full&apikey=" + api_key,
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Fout bij het ophalen van gegevens voor:", symbol)
        return None

# Functie om technische indicatoren te berekenen
def calculate_macd(close_price, window_short=12, window_long=26, window_signal=9):
    ema_short = close_price.ewm(span=window_short, adjust=False).mean()
    ema_long = close_price.ewm(span=window_long, adjust=False).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=window_signal, adjust=False).mean()
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
        print("Buying {}.".format(symbol))
    else:
        print("Selling {}.".format(symbol))

# Hoofdfunctie om geautomatiseerd handelen uit te voeren
def auto_trade(symbol):
    data = fetch_data(symbol)
    if data is not None:
        data = pd.DataFrame(data['Time Series (5min)']).T
        data = calculate_technical_indicators(data)
        
        X = prepare_data(data)
        y = (data['4. close'].shift(-1) > data['4. close']).astype(int).values[:-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

        model = train_model(X_train, y_train)
        signals = generate_signals(data, model)
        execute_trade(symbol, signals)

# Start geautomatiseerd handelen voor elk opgegeven aandeel
symbols = ['STOXX50', 'NDX', 'XAUUSD', 'BTCUSD', 'ETHUSD', 'EURUSD']
iterations = 10
for i in range(iterations):
    for symbol in symbols:
        auto_trade(symbol)
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
    def __init__(self, demo_mode=False):
        self.demo_mode = demo_mode
        if self.demo_mode:
            self.trading_balance = 50000  # Nep balance voor demo-modus
        else:
            self.trading_balance = 0  # Start met een lege balans in de normale modus
    
    def deposit_funds(self, amount):
        if self.demo_mode:
            print("Simulating deposit of {} in demo mode.".format(amount))
            # Voer simulatieacties uit om geld toe te voegen aan de demo-balans
            self.trading_balance += amount
        else:
            print("Depositing {} in real mode.".format(amount))
            # Voer echte acties uit om geld toe te voegen aan de handelsaccount
    
    def withdraw_funds(self, amount):
        if self.demo_mode:
            print("Simulating withdrawal of {} in demo mode.".format(amount))
            # Voer simulatieacties uit om geld op te nemen van de demo-balans
            self.trading_balance -= amount
        else:
            print("Withdrawing {} in real mode.".format(amount))
            # Voer echte acties uit om geld op te nemen van de handelsaccount
    
    def toggle_demo_mode(self):
        self.demo_mode = not self.demo_mode
        if self.demo_mode:
            self.trading_balance = 50000  # Reset de demo-balans naar 50000 euro
        print("Demo mode is now", self.demo_mode)

# Voeg deze code toe na je hoofdfunctionaliteit

# Voorbeeldgebruik van de TradingApp-klasse
app = TradingApp()  # Maak een instantie van de app

# Implementeer de rest van je code hieronder
# Je kunt de TradingApp-methoden gebruiken om stortingen, opnames en schakelen tussen demo-modus en echte modus uit te voeren
