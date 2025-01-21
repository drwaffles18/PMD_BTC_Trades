import pandas as pd
import numpy as np
from tradingview_ta import TA_Handler, Interval, Exchange
from datetime import datetime
import requests


class GetBTCData:
    def __init__(self, symbol, interval, start_date, end_date=None):
        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date

    def get_binance_data(self):
        url = 'https://api.binance.com/api/v3/klines'
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'startTime': int(self.start_date.timestamp() * 1000),
            'limit': 1000  # Máximo permitido por Binance
        }
        if self.end_date:
            params['endTime'] = int(self.end_date.timestamp() * 1000)

        data = []
        while True:
            response = requests.get(url, params=params)
            response.raise_for_status()
            chunk = response.json()
            if not chunk:
                break
            data.extend(chunk)
            params['startTime'] = chunk[-1][0] + 1  # Actualiza startTime para la siguiente solicitud
            
            if self.end_date and params['startTime'] > int(self.end_date.timestamp() * 1000):
                break

        df = pd.DataFrame(data, columns=[
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ])
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
        df['Close'] = pd.to_numeric(df['Close'])
        return df

    def calculate_technical_indicators(self, data):
        data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
        data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['Histogram'] = data['MACD'] - data['Signal_Line']
        return data

    def calculate_stoch_rsi(self, data, rsi_length=14, stoch_length=14, k_period=3, d_period=3):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        avg_gain = gain.rolling(window=rsi_length, min_periods=1).mean()
        avg_loss = loss.rolling(window=rsi_length, min_periods=1).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        rsi_min = data['RSI'].rolling(window=stoch_length, min_periods=1).min()
        rsi_max = data['RSI'].rolling(window=stoch_length, min_periods=1).max()
        data['%K'] = 100 * ((data['RSI'] - rsi_min) / (rsi_max - rsi_min))
        data['%D'] = data['%K'].rolling(window=d_period, min_periods=1).mean()
        return data

    def calculate_additional_indicators(self, data):
        data['MACD Comp'] = np.where(data['MACD'] > data['Signal_Line'], "MACD", "Signal")
        data['Cross Check'] = np.where(data['MACD Comp'] == data['MACD Comp'].shift(1), data['MACD Comp'], data['MACD Comp'] + " Cross")
        data['EMA20 Check'] = data['Close'] > data['EMA20']
        data['EMA50 Check'] = data['Close'] > data['EMA50']
        data['EMA200 Check'] = data['Close'] > data['EMA200']
        data['RSI Check'] = (data['%K'] < 90) & (data['%D'] < 90)

        # Lógica para la señal B-H-S
        conditions = [
            (data['Cross Check'].shift(1) == "MACD Cross") & ((data['EMA20 Check'].shift(1) == True) | (data['EMA50 Check'].shift(1) == True)) & (data['EMA200 Check'].shift(1) == True),
            (data['Cross Check'] == "Signal Cross")
        ]
        choices = ['B', 'S']
        data['B-H-S Signal'] = np.select(conditions, choices, default='H')

        # Asegurarse de que las entradas después del registro 10744 sean NaN o en blanco
        data['B-H-S Signal'] = np.where(data.index <= 10743, data['B-H-S Signal'], np.nan)

        return data

    def get_data(self):
        data = self.get_binance_data()
        data = self.calculate_technical_indicators(data)
        data = self.calculate_stoch_rsi(data)
        data = self.calculate_additional_indicators(data)
        return data

