import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import pytz
import time
from datetime import datetime, timedelta, time
from pandas import Timestamp
import numpy as np


#librerias para series de tiempo

from Prediccion_Vic import HW_calibrado
from Correccion_Fechas import Correccion_Fechas
from tradingview_ta import Interval
from datetime import datetime 
from statsmodels.tsa.api import ExponentialSmoothing

#libreria de prediccion

from Bayes_Buy_Signals import BayesSignalPredictor



def load_trade_signals(sheet_id):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df_signals = pd.read_csv(url)
    df_signals['Open time'] = pd.to_datetime(df_signals['Open time'])
    df_signals['Open time'] = df_signals['Open time'].dt.tz_localize('UTC').dt.tz_convert('America/Costa_Rica')
    return df_signals

def get_binance_data(symbol, interval, start, end=None, sheet_id=None):
    url = 'https://api.binance.us/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': int(start.timestamp() * 1000),
        'limit': 1000  # Máximo permitido por Binance
    }
    if end:
        params['endTime'] = int(end.timestamp() * 1000)

    data = []
    while True:
        response = requests.get(url, params=params)
        response.raise_for_status()
        chunk = response.json()
        if not chunk:
            break
        data.extend(chunk)
        params['startTime'] = chunk[-1][0] + 1  # Actualiza startTime para la siguiente solicitud
        
        if end and params['startTime'] > int(end.timestamp() * 1000):
            break

    df = pd.DataFrame(data, columns=[
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ])

    # Convertir y localizar 'Open time' y 'Close time'
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/Costa_Rica')
    df['Close time'] = pd.to_datetime(df['Close time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/Costa_Rica')

    # Eliminar las dos primeras filas que pertenecen al año 2021
    df = df.drop(df.index[:2])

    # Cargar datos de señales si se proporciona un ID de hoja
    if sheet_id:
        df_signals = load_trade_signals(sheet_id)
        df = df.merge(df_signals, on='Open time', how='left')

    return df

def calculate_stochastic_rsi(df, rsi_length=14, stoch_length=14, k_period=3, d_period=3):
    """Calculate Stochastic RSI."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))

    avg_gain = gain.rolling(window=rsi_length, min_periods=rsi_length).mean()
    avg_loss = loss.rolling(window=rsi_length, min_periods=rsi_length).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Calculate Stochastic RSI
    stoch_rsi = (rsi - rsi.rolling(window=stoch_length, min_periods=stoch_length).min()) / (
            rsi.rolling(window=stoch_length, min_periods=stoch_length).max() - rsi.rolling(window=stoch_length, min_periods=stoch_length).min())
    df['%K'] = stoch_rsi.rolling(window=k_period, min_periods=k_period).mean() * 100
    df['%D'] = df['%K'].rolling(window=d_period, min_periods=d_period).mean()

    return df

def calculate_indicators(df):
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    # EMAs
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal_Line']

    # RSI calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Stochastic RSI
    df = calculate_stochastic_rsi(df)

    # Additional checks
    df['MACD Comp'] = np.where(df['MACD'] > df['Signal_Line'], 'MACD', 'Signal')
    df['Cross Check'] = df['MACD Comp'] != df['MACD Comp'].shift(1)
    df['Cross Check'] = np.where(df['Cross Check'], df['MACD Comp'] + " Cross", df['MACD Comp'])
    df['EMA20 Check'] = (df['Close'] > df['EMA20']).astype(int)
    df['EMA50 Check'] = (df['Close'] > df['EMA50']).astype(int)
    df['EMA 200 Check'] = (df['Close'] > df['EMA200']).astype(int)
    df['RSI Check'] = ((df['%K'] < 90) & (df['%D'] < 90)).astype(int)

    return df

def plot_candlestick_macd_stoch(data):
    if data.empty:
        return None  # Retorna None si no hay datos para evitar errores 
    
    #"""
    #Genera un gráfico de velas con EMAs, MACD y Stochastic RSI como subgráficos.

    #Parameters:
    #- data (DataFrame): DataFrame con columnas requeridas:
    #  'Open time', 'Open', 'High', 'Low', 'Close', 'EMA20', 'EMA50', 'EMA200',
    #  'MACD', 'Signal_Line', 'Histogram', '%K', '%D'.
    #"""

    pio.templates.default = "plotly_dark"

    # Identificar cruces
    cruz_arriba = (data['MACD'].shift(1) < data['Signal_Line'].shift(1)) & (data['MACD'] > data['Signal_Line'])
    cruz_abajo = (data['MACD'].shift(1) > data['Signal_Line'].shift(1)) & (data['MACD'] < data['Signal_Line'])

    cruz_arriba_indices = data[cruz_arriba].index
    cruz_abajo_indices = data[cruz_abajo].index



    # Crear subgráficos
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],  # Alturas relativas de los subgráficos
        vertical_spacing=0.02,
        subplot_titles=("Gráfico de Velas con EMAs", "MACD con Histograma", "Stochastic RSI")
    )

    # Subgráfico 1: Gráfico de velas con EMAs
    fig.add_trace(
        go.Candlestick(
            x=data['Open time'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            increasing_line_color='rgba(0, 250, 0, 0.8)',
            decreasing_line_color='rgba(255, 0, 0, 0.8)',
            name='Candlestick'
        ),
        row=1, col=1
    )

    # Añadir EMAs
    fig.add_trace(go.Scatter(x=data['Open time'], y=data['EMA20'], mode='lines', line=dict(color='blue'), name='EMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Open time'], y=data['EMA50'], mode='lines', line=dict(color='green'), name='EMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Open time'], y=data['EMA200'], mode='lines', line=dict(color='yellow'), name='EMA 200'), row=1, col=1)

    # Añadir anotaciones para señales de compra y venta
    buys = data[data['B-H-S Signal'] == 'B']['Open time']
    sells = data[data['B-H-S Signal'] == 'S']['Open time']

    for buy in buys:
        high_price = data.loc[data['Open time'] == buy, 'High'].max()
        fig.add_trace(go.Scatter(
            x=[buy],
            y=[high_price],
            text=["Buy"],
            mode="markers+text",
            marker=dict(color='green', size=10, symbol='triangle-up'),
            name='Buy Signal',
            showlegend=False,
            hoverinfo='skip',
            textposition="top center"
        ), row=1, col=1)

    for sell in sells:
        low_price = data.loc[data['Open time'] == sell, 'Low'].min()
        fig.add_trace(go.Scatter(
            x=[sell],
            y=[low_price],
            text=["Sell"],
            mode="markers+text",
            marker=dict(color='red', size=10, symbol='triangle-down'),
            name='Sell Signal',
            showlegend=False,
            hoverinfo='skip',
            textposition="bottom center"
        ), row=1, col=1)



    # Subgráfico 2: MACD con histograma
    fig.add_trace(go.Bar(x=data['Open time'], y=data['Histogram'],name='Histograma',marker_color=['rgba(173, 216, 230, 0.8)' if h > 0 else 'red' for h in data['Histogram']]),row=2, col=1)
    # Marcadores de cruces

    fig.add_trace(go.Scatter(x=data.loc[cruz_arriba_indices, 'Open time'], y=data.loc[cruz_arriba_indices, 'MACD'],mode='markers',marker=dict(color='green', size=10),name='Cruce hacia arriba'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.loc[cruz_abajo_indices, 'Open time'],y=data.loc[cruz_abajo_indices, 'MACD'],mode='markers',marker=dict(color='red', size=10),name='Cruce hacia abajo'), row=2, col=1)
    
    #lineas MACD y Signal
    fig.add_trace(go.Scatter(x=data['Open time'], y=data['MACD'], mode='lines', line=dict(color='green'), name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data['Open time'], y=data['Signal_Line'], mode='lines', line=dict(color='orange'), name='Signal Line'), row=2, col=1)

    # Subgráfico 3: Stochastic RSI
    # Agregar bandas (80, 50, 20)
    fig.add_shape(type="rect",xref="x", yref="y3",x0=min(data['Open time']), x1=max(data['Open time']),y0=20, y1=80,fillcolor="gray", opacity=0.2, layer="below", line_width=0    )
    fig.add_trace(go.Scatter(x=data['Open time'], y=data['%K'], mode='lines', line=dict(color='blue'), name='%K'), row=3, col=1)
    fig.add_trace(go.Scatter(x=data['Open time'], y=data['%D'], mode='lines', line=dict(color='orange'), name='%D'), row=3, col=1)

    # Configurar layout
    fig.update_layout(
        title='Gráfico de Velas con EMAs, MACD y Stochastic RSI',
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5, font=dict(color="white")),
        height=800,
        width= 1500,
        plot_bgcolor='black',  # Fondo del gráfico
        paper_bgcolor='black',  # Fondo alrededor del gráfico
        font_color="white",  # Color del texto
        title_font_color='white', 
        xaxis=dict(
            showspikes=True,  # Mostrar líneas de crosshair
            spikemode="across",  # Líneas horizontales y verticales
            spikesnap="cursor",  # Los spikes se ajustan al cursor
            showline=True,
            showgrid=True
        ),
        hovermode="x unified"  # Mostrar todos los datos de un punto vertical
    )

    # Configurar ejes para cada subgráfico
    fig.update_yaxes(title_text="Precio", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="Stochastic RSI", row=3, col=1, range=[0, 100])

    # Desactivar el rangeslider
    fig.update_xaxes(rangeslider=dict(visible=False))

    return fig  # Retorna la figura para que sea usada en Streamlit

def preparar_serie_tiempo(data):
    columns_to_keep = ['Open time', 'Close']
    btc_data_time = data[columns_to_keep]
    correccion = Correccion_Fechas(btc_data_time, 'Open time', freq='4h')
    correccion.identificar_fechas_faltantes()
    correccion.agregar_fechas_faltantes()
    correccion.imputar_datos_suavizados('Close', periodos=5)
    btc_ts = correccion.convertir_a_serie_tiempo('Close')
    return btc_ts


def agregar_prediccion(fig, btc_ts):
    modelo_hw = ExponentialSmoothing(btc_ts, trend='add', seasonal='add', seasonal_periods=4)
    fitted_model = modelo_hw.fit(smoothing_level=0.3, smoothing_slope=0.1, smoothing_seasonal=0.7)
    pred_hw = fitted_model.forecast(24 * 3)  # 3 días de predicción

    last_date = btc_ts.index[-1]
    pred_dates = pd.date_range(start=last_date, periods=len(pred_hw) + 1, freq='4H')[1:]

    # Asegúrate de que los pred_dates son datetime y no contienen nulos
    if pred_dates.isnull().any() or pred_hw.isnull().any():
        st.error("Error en los datos de predicción. Revisa las fechas y los valores.")
        return fig

    # Añadir la traza de predicción
    fig.add_trace(go.Scatter(x=pred_dates, y=pred_hw, mode='lines', line=dict(color='cyan', width=2), name='Predicción HW'), row=1, col=1)
    return fig

def convert_and_format(df):
    # Lista de columnas que necesitan ser convertidas y formateadas
    columns_to_convert = [
        'Open', 'High', 'Low', 'Volume',
        'Quote asset volume', 'Taker buy base asset volume', 'Taker buy quote asset volume'
    ]

    # Convertir columnas a numérico, ignorando errores para evitar problemas con datos no convertibles
    for col in columns_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Formatear números a dos decimales
    format_dict = {col: "{:.2f}" for col in columns_to_convert}

    # Aplicar formateo de números y estilo condicional
    df_styled = df.style.format(format_dict).apply(highlight_signals, axis=1)
    return df_styled

def highlight_signals(row):
    if row['B-H-S Signal'] == 'B':
        return ['background-color: lightgreen'] * len(row)
    elif row['B-H-S Signal'] == 'S':
        return ['background-color: lightcoral'] * len(row)
    else:
        return [''] * len(row)  # sin color para las demas filas

def app():
    st.sidebar.title("Opciones de Visualización")
    
    tz = pytz.timezone('America/Costa_Rica')
    symbol = 'BTCUSDT'
    interval = '4h'
    sheet_id = '1SSNJk-NeOgCUoUZ-CU5J3HiZYb1UDcCuqs8U3NhKOqs'
    start_date = datetime(2022, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime.now(tz) + timedelta(hours=24)
    
    # Cargar y procesar datos
    full_data = get_binance_data(symbol, interval, start_date, end_date, sheet_id=sheet_id)
    full_data = calculate_indicators(full_data)
    full_data_styled = convert_and_format(full_data)
    
    # Formatear decimales
    format_dict = {
        'Open': "{:.2f}",
        'High': "{:.2f}",
        'Low': "{:.2f}",
        'Volume': "{:.2f}",
        'Quote asset volume': "{:.2f}",
        'Taker buy base asset volume': "{:.2f}",
        'Taker buy quote asset volume': "{:.2f}"
    }
    full_data_styled = full_data.style.format(format_dict)

    # Definir filtros globales
    filtered_start_date = st.sidebar.date_input("Inicio", start_date.date())
    filtered_end_date = st.sidebar.date_input("Fin", datetime.now(tz).date())
    st.sidebar.text("Seleccione que deseas hacer.")
    filtered_start_date = pd.Timestamp(filtered_start_date, tz='America/Costa_Rica')
    filtered_end_date = pd.Timestamp(filtered_end_date, tz='America/Costa_Rica')
    mask = (full_data['Open time'] >= filtered_start_date) & (full_data['Open time'] <= filtered_end_date)

    # Convertir y formatear datos
    full_data_styled = convert_and_format(full_data)

    # Crear pestañas
    tab1, tab2 = st.tabs(["Visualización", "Datos Históricos"])

    # Pestaña de Visualización
    with tab1:
        if st.sidebar.button("Mostrar Gráfico 4H BTC"):
            filtered_data = full_data.loc[mask]
            fig = plot_candlestick_macd_stoch(filtered_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                # Guardar el gráfico y los datos filtrados en el estado de la sesión
                st.session_state['current_fig'] = fig
                st.session_state['filtered_data'] = filtered_data
            else:
                st.error("No se pudo generar el gráfico. Por favor, revisa el rango de fechas seleccionado.")
        
        if st.sidebar.button("Agregar Forecast de Precio") and 'current_fig' in st.session_state and 'filtered_data' in st.session_state:
            btc_ts = preparar_serie_tiempo(st.session_state['filtered_data'])
            fig = agregar_prediccion(st.session_state['current_fig'], btc_ts)
            st.plotly_chart(fig, use_container_width=True)
        
        # Botón para generar predicciones de señales de compra/venta
        if st.sidebar.button("Generar Predicción Buy/Sell"):
            predictor = BayesSignalPredictor()  
            full_data_with_predictions = predictor.predict_signals(full_data)
            if 'current_fig' in st.session_state:
                # Actualizar el gráfico existente con nuevas señales
                updated_fig = plot_candlestick_macd_stoch(full_data_with_predictions.loc[mask])
                st.plotly_chart(updated_fig, use_container_width=True)
            else:
                st.error("Primero genera el gráfico para poder añadir predicciones.")

    # Pestaña de Datos Históricos
    with tab2:
        st.write("Datos Históricos")
        full_data_styled = convert_and_format(full_data) 
        st.dataframe(full_data_styled)

if __name__ == "__main__":
    app()

