import warnings
import statistics
import pandas as pd
from abc import  ABCMeta, abstractmethod
from statsmodels.tsa.api import ExponentialSmoothing
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from joblib import Parallel, delayed
from statsmodels.tsa.statespace.sarimax import SARIMAX


class BasePrediccion(metaclass = ABCMeta):    
  @abstractmethod
  def forecast(self):
    pass

class Prediccion(BasePrediccion):
  def __init__(self, modelo):
    self.__modelo = modelo
  
  @property
  def modelo(self):
    return self.__modelo  
  
  @modelo.setter
  def modelo(self, modelo):
    if(isinstance(modelo, Modelo)):
      self.__modelo = modelo
    else:
      warnings.warn('El objeto debe ser una instancia de Modelo.')
  
class meanfPrediccion(Prediccion):
  def __init__(self, modelo):
    super().__init__(modelo)
  
  def forecast(self, steps = 1):
    res = []
    for i in range(steps):
      res.append(self.modelo.coef)
    
    start = self.modelo.ts.index[-1]
    freq  = self.modelo.ts.index.freqstr
    fechas = pd.date_range(start = start, periods = steps+1, freq = freq)
    fechas = fechas.delete(0)
    res = pd.Series(res, index = fechas)
    return(res)

class naivePrediccion(Prediccion):
  def __init__(self, modelo):
    super().__init__(modelo)
  
  def forecast(self, steps = 1):
    res = []
    for i in range(steps):
      res.append(self.modelo.coef)
    
    start = self.modelo.ts.index[-1]
    freq  = self.modelo.ts.index.freqstr
    fechas = pd.date_range(start = start, periods = steps+1, freq = freq)
    fechas = fechas.delete(0)
    res = pd.Series(res, index = fechas)
    return(res)

class snaivePrediccion(Prediccion):
  def __init__(self, modelo):
    super().__init__(modelo)
  
  def forecast(self, steps = 1):
    res = []
    pos = 0
    for i in range(steps):
      if pos >= len(self.modelo.coef):
        pos = 0
      res.append(self.modelo.coef[pos])
      pos = pos + 1
    
    start = self.modelo.ts.index[-1]
    freq  = self.modelo.ts.index.freqstr
    fechas = pd.date_range(start = start, periods = steps+1, freq = freq)
    fechas = fechas.delete(0)
    res = pd.Series(res, index = fechas)
    return(res)

class driftPrediccion(Prediccion):
  def __init__(self, modelo):
    super().__init__(modelo)
  
  def forecast(self, steps = 1):
    res = []
    for i in range(steps):
      res.append(self.modelo.ts[-1] + self.modelo.coef * i)
    
    start = self.modelo.ts.index[-1]
    freq  = self.modelo.ts.index.freqstr
    fechas = pd.date_range(start = start, periods = steps+1, freq = freq)
    fechas = fechas.delete(0)
    res = pd.Series(res, index = fechas)
    return(res)

class BaseModelo(metaclass = ABCMeta):    
  @abstractmethod
  def fit(self):
    pass

class Modelo(BaseModelo):
  def __init__(self, ts):
    self.__ts = ts
    self._coef = None
  
  @property
  def ts(self):
    return self.__ts  
  
  @ts.setter
  def ts(self, ts):
    if(isinstance(ts, pd.core.series.Series)):
      if(ts.index.freqstr != None):
        self.__ts = ts
      else:
        warnings.warn('ERROR: No se indica la frecuencia de la serie de tiempo.')
    else:
      warnings.warn('ERROR: El parámetro ts no es una instancia de serie de tiempo.')
  
  @property
  def coef(self):
    return self._coef
  
class meanf(Modelo):
  def __init__(self, ts):
    super().__init__(ts)
  
  def fit(self):
    self._coef = statistics.mean(self.ts)
    res = meanfPrediccion(self)
    return(res)

class naive(Modelo):
  def __init__(self, ts):
    super().__init__(ts)
  
  def fit(self):
    self._coef = self.ts[-1]
    res = naivePrediccion(self)
    return(res)

class snaive(Modelo):
  def __init__(self, ts):
    super().__init__(ts)
  
  def fit(self, h = 1):
    self._coef = self.ts.values[-h:]
    res = snaivePrediccion(self)
    return(res)

class drift(Modelo):
  def __init__(self, ts):
    super().__init__(ts)
  
  def fit(self):
    self._coef = (self.ts[-1] - self.ts[0]) / len(self.ts)
    res = driftPrediccion(self)
    return(res)
#holt-winters

class HW_Prediccion(Prediccion):
  def __init__(self, modelo, alpha, beta, gamma):
    super().__init__(modelo)
    self.__alpha = alpha
    self.__beta  = beta
    self.__gamma = gamma
  
  @property
  def alpha(self):
    return self.__alpha
  
  @property
  def beta(self):
    return self.__beta 
  
  @property
  def gamma(self):
    return self.__gamma
  
  def forecast(self, steps = 1):
    res = self.modelo.forecast(steps)
    return(res)

class HW_calibrado(Modelo):
    def __init__(self, ts, test, trend='add', seasonal='add'):
        super().__init__(ts)
        self.__test = test
        self.__modelo = ExponentialSmoothing(ts, trend=trend, seasonal=seasonal)

    def _fit_single_model(self, alpha, beta, gamma):
        try:
            model_fit = self.__modelo.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
            pred = model_fit.forecast(len(self.__test))
            mse = ((pred - self.__test) ** 2).mean()
            return (mse, model_fit, alpha, beta, gamma)
        except Exception as e:
            return (float('inf'), None, alpha, beta, gamma)

    def fit(self, paso=0.1):
        n = np.append(np.arange(0, 1, paso), 1)
        results = Parallel(n_jobs=-1)(delayed(self._fit_single_model)(alpha, beta, gamma)
                                      for alpha in n for beta in n for gamma in n)

        best_mse = float('inf')
        best_model = None
        best_params = (0.1, 0.1, 0.1)  # Default values if no better model is found

        for mse, model, alpha, beta, gamma in results:
            if mse < best_mse:
                best_mse = mse
                best_model = model
                best_params = (alpha, beta, gamma)

        if best_model is not None:
            return HW_Prediccion(best_model, *best_params)
        else:
            raise ValueError("No se pudo mejorar el modelo inicial. Revise los datos y la configuración del modelo.")


##deep learning
class LSTM_TSPrediccion(Prediccion):
  def __init__(self, modelo):
    super().__init__(modelo)
    self.__scaler = MinMaxScaler(feature_range = (0, 1))
    self.__X = self.__scaler.fit_transform(self.modelo.ts.to_frame())
  
  def __split_sequence(self, sequence, n_steps):
    X, y = [], []
    for i in range(n_steps, len(sequence)):
      X.append(self.__X[i-n_steps:i, 0])
      y.append(self.__X[i, 0])
    return np.array(X), np.array(y)
  
  def forecast(self, steps = 1):
    res = []
    p = self.modelo.p
    for i in range(steps):
      y_pred = [self.__X[-p:]]
      X, y = self.__split_sequence(self.__X, p)
      X = np.reshape(X, (X.shape[0], X.shape[1], 1))
      self.modelo.m.fit(X, y, epochs = 10, batch_size = 1, verbose = 0)
      pred = self.modelo.m.predict(y_pred)
      res.append(self.__scaler.inverse_transform(pred).tolist()[0][0])
      self.__X = np.append(self.__X, pred.tolist(), axis = 0)
    
    start  = self.modelo.ts.index[-1]
    freq   = self.modelo.ts.index.freqstr
    fechas = pd.date_range(start = start, periods = steps+1, freq = freq)
    fechas = fechas.delete(0)
    res = pd.Series(res, index = fechas)
    return(res)

class LSTM_TS(Modelo):
  def __init__(self, ts, p = 1, lstm_units = 50, dense_units = 1, optimizer = 'rmsprop', loss = 'mse'):
    super().__init__(ts)
    self.__p = p
    self.__m = Sequential()
    self.__m.add(LSTM(units = lstm_units, input_shape = (p, 1)))
    self.__m.add(Dense(units = dense_units))
    self.__m.compile(optimizer = optimizer, loss = loss)
  
  @property
  def m(self):
    return self.__m
  
  @property
  def p(self):
    return self.__p
  
  def fit(self):
    res = LSTM_TSPrediccion(self)
    return(res)

class SARIMA_Prediccion(Prediccion):
    def __init__(self, modelo, p, d, q, P, D, Q, S):
        super().__init__(modelo)
        self.__p = p
        self.__d = d
        self.__q = q
        self.__P = P
        self.__D = D
        self.__Q = Q
        self.__S = S

    @property
    def p(self):
        return self.__p

    @property
    def d(self):
        return self.__d

    @property
    def q(self):
        return self.__q

    @property
    def P(self):
        return self.__P

    @property
    def D(self):
        return self.__D

    @property
    def Q(self):
        return self.__Q

    @property
    def S(self):
        return self.__S

    def forecast(self, steps=1):
        res = self.modelo.forecast(steps)
        return res

class SARIMA_calibrado(Modelo):
    def __init__(self, ts, test):
        super().__init__(ts)
        self.__test = test

    @property
    def test(self):
        return self.__test  

    @test.setter
    def test(self, test):
        if isinstance(test, pd.Series):
            if test.index.freqstr is not None:
                self.__test = test
            else:
                warnings.warn('ERROR: No se indica la frecuencia de la serie de tiempo.')
        else:
            warnings.warn('ERROR: El parámetro test no es una instancia de serie de tiempo.')

    def fit(self, max_p=2, max_d=2, max_q=2, max_P=1, max_D=1, max_Q=1, S=None, n_jobs=-1):
      if S is None:
          warnings.warn('ERROR: No se indica el periodo a utilizar (S).')
          return None
      
      # Crear los rangos según los límites
      ar = range(0, max_p + 1)
      es = range(0, max_P + 1)

      # Parámetros de búsqueda en cuadrícula
      param_grid = [(p, d, q, P, D, Q) 
                    for p in ar 
                    for d in range(0, max_d + 1) 
                    for q in range(0, max_q + 1) 
                    for P in es 
                    for D in range(0, max_D + 1) 
                    for Q in range(0, max_Q + 1)]
      
      error = float("inf")
      best_params = None
      best_model_fit = None

      # Función para evaluar cada combinación de parámetros
      def evaluate_params(params):
          p, d, q, P, D, Q = params
          try:
              with warnings.catch_warnings():
                  warnings.simplefilter("ignore")
                  modelo = SARIMAX(self.ts, order=[p, d, q], seasonal_order=[P, D, Q, S])
                  model_fit = modelo.fit(disp=False)
                  pred = model_fit.forecast(len(self.test))
                  mse = np.sum((pred - self.test) ** 2)
              return mse, params, model_fit
          except:
              return float("inf"), None, None
      
      # Ejecutar evaluaciones en paralelo
      results = Parallel(n_jobs=n_jobs)(delayed(evaluate_params)(params) for params in param_grid)

      # Filtrar el mejor resultado
      for mse, params, model_fit in results:
          if mse < error:
              error = mse
              best_params = params
              best_model_fit = model_fit

      # Asignar valores óptimos y devolver el modelo
      if best_params:
          res_p, res_d, res_q, res_P, res_D, res_Q = best_params
          return SARIMA_Prediccion(best_model_fit, res_p, res_d, res_q, res_P, res_D, res_Q, S)
      else:
          warnings.warn("No se encontró un modelo válido.")
          return None

