from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import pickle

class BayesSignalPredictor:
    def __init__(self, model_path='C:\\Users\\chefv\\OneDrive\\Documents\\PROMiDAT\\Inteligencia Artificial\\Proyecto Final\\best_model_bayes.pkl'):
        # Cargar el modelo de Bayes
        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)
        self.scaler = StandardScaler()
        self.label_encoders = {
            'MACD Comp': LabelEncoder(),
            'Cross Check': LabelEncoder()
        }
        self.initialized = False

    def prepare_data(self, df):
        # Filtrar el DataFrame para solo las columnas necesarias, excluyendo 'B-H-S Signal'
        columns_to_keep = ['EMA20', 'EMA50', 'EMA200', 'EMA_12', 'EMA_26', 'MACD', 'Signal_Line', 'RSI', '%K', '%D', 
                           'MACD Comp', 'Cross Check', 'EMA20 Check', 'EMA 200 Check', 'RSI Check']
        
        df = df[columns_to_keep]

        # Ajustar el scaler y transformar las columnas numéricas
        features_to_scale = ['EMA20', 'EMA50', 'EMA200', 'EMA_12', 'EMA_26', 'MACD', 'Signal_Line', 'RSI', '%K', '%D']
        if not self.initialized:
            self.scaler.fit(df[features_to_scale])
            self.initialized = True
        df[features_to_scale] = self.scaler.transform(df[features_to_scale])

        # Codificar las columnas categóricas
        df['MACD Comp'] = self.label_encoders['MACD Comp'].fit_transform(df['MACD Comp'])
        df['Cross Check'] = self.label_encoders['Cross Check'].fit_transform(df['Cross Check'])

        return df

    def predict_signals(self, df):
        # Filtrar para obtener solo las filas sin señales
        data_to_predict = df[df['B-H-S Signal'].isna()]
        if not data_to_predict.empty:
            # Preparar los datos
            prepared_data = self.prepare_data(data_to_predict)
            # Realizar predicciones
            predictions = self.model.predict(prepared_data)
            df.loc[df['B-H-S Signal'].isna(), 'B-H-S Signal'] = predictions
        return df
