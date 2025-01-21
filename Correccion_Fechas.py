import pandas as pd

class Correccion_Fechas:
    def __init__(self, dataframe, fecha_col, freq='MS'):
        self._df = dataframe
        self._fecha_col = fecha_col
        self._freq = freq
        self._faltan_fechas = None

    @property
    def dataframe(self):
        return self._df

    @dataframe.setter
    def dataframe(self, value):
        self._df = value

    @property
    def fecha_col(self):
        return self._fecha_col

    @fecha_col.setter
    def fecha_col(self, value):
        self._fecha_col = value

    @property
    def faltan_fechas(self):
        return self._faltan_fechas

    def identificar_fechas_faltantes(self):
        """Identifica las fechas faltantes en el dataframe."""
        self._df[self._fecha_col] = pd.to_datetime(self._df[self._fecha_col])
        total_fechas = pd.date_range(start=self._df[self._fecha_col].min(), end=self._df[self._fecha_col].max(), freq=self._freq)
        self._faltan_fechas = total_fechas.difference(self._df[self._fecha_col])
        if self._faltan_fechas.empty:
            print("No hay fechas faltantes.")
        else:
            print("Fechas faltantes:", self._faltan_fechas)
        return self._faltan_fechas

    def agregar_fechas_faltantes(self):
        """Agrega las fechas faltantes al dataframe asegurando la correcta frecuencia."""
        if self._faltan_fechas is not None and not self._faltan_fechas.empty:
            df_temporal = pd.DataFrame(index=self._faltan_fechas)
            df_temporal = df_temporal.reindex(columns=self._df.columns, fill_value=pd.NA)
            df_temporal[self._fecha_col] = df_temporal.index
            self._df = pd.concat([self._df, df_temporal], ignore_index=False)
            self._df.sort_values(by=[self._fecha_col], inplace=True)
            self._df.set_index(self._fecha_col, inplace=True, drop=False)
        else:
            print("No se agregaron fechas faltantes porque no hay ninguna.")

    def imputar_datos_suavizados(self, columna, periodos=5):
        """Imputa los datos faltantes usando un suavizado. Asegura que NaNs no rompan el proceso."""
        # Rellena NaNs temporalmente para la imputación
        self._df[columna].fillna(method='ffill', inplace=True)
        self._df[columna].fillna(method='bfill', inplace=True)

        self._df[columna] = self._df[columna].rolling(window=periodos, min_periods=1, center=True).mean()

        # Opcional: Restaurar NaN donde originalmente eran NaN si es necesario
        print("Datos suavizados correctamente.")

    def convertir_a_serie_tiempo(self, valor_col):
        """Convierte el dataframe en una serie de tiempo manteniendo los tiempos originales."""
        if self._fecha_col in self._df.columns:
            self._df.set_index(self._fecha_col, inplace=True, drop=True)
        else:
            self._df.reset_index(inplace=True)
        fechas = pd.DatetimeIndex(self._df.index)  # Usar el índice directamente si ya está configurado
        serie_tiempo = pd.Series(self._df[valor_col].values, index=fechas)
        return serie_tiempo


    def __str__(self):
        return f"Correccion_Fechas(DataFrame con {len(self._df)} filas, columna de fechas: '{self._fecha_col}', frecuencia: '{self._freq}')"
