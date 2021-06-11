import numpy as np
import statistics
import tensorflow as tf
import sklearn

from tensorflow import keras
from sklearn.metrics import mean_squared_error

class ModeloLSTM():
    
    def __init__(self, input_shape, name="LSTM", shape='many-to-one', step=None):
        """
        Descrição:
        ----------
        Construtor da classe 'ModeloLSTM'

        Parâmetros:
        -----------
        input_dim: tuple
            Formato de entrada do preditor
        step: int
            Passo de predição. Somente utilizado caso o formato de entrada for 'many-to-many'
        shape: str
            Forma do vetor de entrada ('many-to-many', 'one-to-one' ou 'many-to-one')    
        name: str
            Nome a ser dado para o modelo

        Retorna:
        --------
        Nada
        """
        
        if not (type(input_shape) is tuple):
            raise TypeError("O formato de entrada deve ser uma tupla!")
            
        if not (type(name) is str):
            raise TypeError("O nome do modelo deve ser uma string!")

        if not (type(shape) is str):
            raise TypeError("O formato do vetor de entrada deve ser uma string!")
        
        if not ((shape == 'many-to-one') or (shape == 'one-to-one') or (shape == 'many-to-many')):
            raise ValueError("Formato de vetor de entrada não reconhecido")

        if not ((type(step) is int) or (type(step) is None)):
            raise TypeError("Formato do passo de predição inválido!")

        self._input_shape = input_shape
        self._name = name
        self._shape = shape
        self._step = step
        self.modelo = None
        pass
        
    def criar_modelo(self, n_units=30, init_mode='glorot_uniform'):
        """
        Descrição:
        ----------
        Função para gerar a rede neural LSTM com os parâmetros especificados.
        A função de ativação é a tangente hiperbólica.
        Ainda não foi implementada.
        
        Parâmetros:
        -----------
        
        Retorna:
        --------
        Nada
        """

        if not (type(n_units) is int):
            raise TypeError("O número de unidades LSTM deve ser um int!")

        if not (type(init_mode) is str):
            raise TypeError("A inicialização deve ser uma string!")              
        
        # dimensoes de entrada
        input_shape = self._input_shape
        shape = self._shape
        
        if (shape == 'many-to-many'):
            return_sequences = True
            step = self._step
        else:
            return_sequences = False

        # nome da rede
        name = self._name
    
        model = keras.Sequential(name=name)
        model.add(keras.Input(shape=input_shape))
        model.add(keras.layers.LSTM(n_units, activation='tanh',
                                    kernel_initializer=init_mode, return_sequences=return_sequences,
                                    name="camada_lstm"))
        if (shape == 'many-to-many'):
            model.add(keras.layers.Dense(step, activation='linear', name="camada_de_saida"))
        else:
            model.add(keras.layers.Dense(1, activation='linear', name="camada_de_saida"))
        
        self._modelo = model
        pass

    
    def gridsearch(self, init_mode='glorot_uniform', n_units=30, 
                   optimizer='Nadam', learning_rate=0.001, loss="mean_squared_error"):
        """
        Descrição:
        ----------
        Método utilizado para o gridsearch da LSTM
        Essa função não altera o modelo do objeto, é apenas para o Gridsearch!
        
        Parâmetros:
        -----------
        init_mode: str
            Inicialização a ser utilizada para o ajuste dos pesos dos neurônios
        n_units: int
            Número de unidades LSTM a ser utilizado na camada das células recorrentes  
        optimizer: str
            Otimizador a ser utilizado
        learning_rate: float
            Taxa de aprendizagem
        loss: str
            Função custo

        Retorna:
        --------
        Um modelo já compilado para ser fornecido ao KerasRegressor
        """

        if not (type(init_mode) is str):
            raise TypeError("A inicialização deve ser uma string!")              
        
        if not (type(n_units) is int):
            raise TypeError("O número de unidades LSTM deve ser um int!")
            
        if not(type(optimizer) is str):
            raise TypeError("O otimizador deve ser uma string!")
            
        if not(type(learning_rate) is float):
            raise TypeError("O learning rate deve ser um float!")
            
        if not(type(loss) is str):
            raise TypeError("A função custo deve ser uma string!")     
            
        # dimensoes de entrada
        input_shape = self._input_shape
        
        # nome da rede
        name = self._name
    
        model = keras.Sequential(name=name)
        model.add(keras.Input(shape=input_shape))
        model.add(keras.layers.LSTM(n_units, activation='tanh', kernel_initializer=init_mode, name="camada_lstm"))
        model.add(keras.layers.Dense(1, activation='linear', name="camada_de_saida"))
    
        # define o otimizador e learning rate
        if (optimizer=='Nadam'):
            model_optimizer = keras.optimizers.Nadam()
        else:
            model_optimizer = optimizer    
        model_optimizer.learning_rate.assign(learning_rate)
    
        model.compile(
            optimizer = model_optimizer,
            loss = 'mse')
    
        model.build()
        return model
    
    def montar(self, optimizer='Nadam', learning_rate=0.001, loss="mean_squared_error"):
        """
        Descrição:
        ----------
        Função para compilar e montar o modelo com o otimizador selecionado com o learning rate escolhido
        e adotando a função custo parametrizada
        
        Parâmetros:
        -----------
        optimizer: str
            Otimizador a ser utilizado
        learning_rate: float
            Taxa de aprendizagem
        loss: str
            Função custo
        
        Retorna:
        --------
        Nada
        """
        
        if not(type(optimizer) is str):
            raise TypeError("O otimizador deve ser uma string!")
            
        if not(type(learning_rate) is float):
            raise TypeError("O learning rate deve ser um float!")
            
        if not(type(loss) is str):
            raise TypeError("A função custo deve ser uma string!")
    
        modelo = self._modelo
        
        if (optimizer=='Nadam'):
            model_optimizer = keras.optimizers.Nadam()
        else:
            model_optimizer = optimizer

        model_optimizer.learning_rate.assign(learning_rate)    
            
        modelo.compile(optimizer = model_optimizer,
                       loss = loss)

        modelo.build()
        
        self._modelo = modelo
        pass
    
    def visualizar(self):
        """
        Descrição:
        ----------
        Função para visualizar a rede neural com o summary do keras
        
        Parâmetros:
        -----------
        Nenhuma
        
        Retorna:
        --------
        Um sumário contendo os parâmetros da rede neural
        """
        
        return self._modelo.summary()
    
    def treinar(self, X_treino, X_val, y_treino, y_val, batch_size=10, early_stopping="ON", epochs=100):
        """
        Descrição:
        ----------
        Função para realizar o treinamento da rede

        Por padrão, essa função formata os dados em Many-to-One
        
        Parâmetros:
        -----------
        X_treino: np.ndarray
            Conjunto de entradas para o treinamento
        X_val: np.ndarray
            Conjunto de entradas para a validação
        y_treino: np.ndarray
            Conjunto de saidas para o treinamento
        y_val: np.ndarray
            Conjunto de saídas para a validação
        batch_size: int
            Tamanho do batch a ser utilizado durante o treinamento
        early_stopping: str
            Se deve "ON" ou não deve "OFF" utilizar early stopping
        epochs: int
            Número de épocas para o treinamento

        Retorna:
        --------
        Nada
        """
        
        if not (type(X_treino) is np.ndarray):
            raise TypeError("Os dados de entrada de treino devem ser um array do numpy!")  
            
        if not (type(X_val) is np.ndarray):
            raise TypeError("Os dados de entrada de validação devem ser um array do numpy!")  
            
        if not (type(y_treino) is np.ndarray):
            raise TypeError("Os dados de saída de treino devem ser um array do numpy!")  
            
        if not (type(y_val) is np.ndarray):
            raise TypeError("Os dados de saída de validação devem ser um array do numpy!")  
            
        if not (type(batch_size) is int):
            raise TypeError("O batch size deve ser um inteiro!")
            
        if not (type(early_stopping) is str):
            raise TypeError("O parâmetro de early stopping deve ser uma string!")

        if not ((early_stopping == 'ON') | (early_stopping == 'OFF')):
            raise ValueError("A string do early stopping deve ser um ON ou OFF")            
            
        if not (type(epochs) is int):
            raise TypeError("O número de épocas deve ser um int!")
            
        modelo = self._modelo
        shape = self._shape

        # formatando os dados de entrada
        len_treino = X_treino.shape[0]
        len_val = X_val.shape[0]
        n_samples = X_treino.shape[1]

        # formatando os dados de entrada para o many-to-one
        if (shape == 'many-to-one'):
            X_treino = np.reshape(X_treino,(len_treino, n_samples, 1))
            X_val = np.reshape(X_val,(len_val, n_samples, 1))

        # formatando os dados de entrada para o one-to-one
        elif (shape == 'one-to-one'):
            X_treino = np.reshape(X_treino,(len_treino, 1, n_samples))
            X_val = np.reshape(X_val,(len_val, 1, n_samples))
        
        if (early_stopping == 'ON'):
            early_stopping_fit = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss')
        else:
            early_stopping_fit = None
        
        modelo.fit(X_treino, y_treino, 
                   epochs=epochs, callbacks=early_stopping_fit,
                   validation_data=(X_val, y_val), batch_size=batch_size,
                   verbose=1)
        
        self._modelo = modelo
        pass
    
    def predicao(self, X_teste, scaler=None):
        """
        Descrição:
        ----------
        Função para predizer os próximos valores utilizando o conjunto de teste

        Por padrão, essa função formata os dados em Many-to-One

        Se os dados passaram por um scaler, passar ele como entrada
        
        Parâmetros:
        -----------
        X_teste: np.ndarray
            Conjunto de entradas para os dados de teste
        scaler: sklearn.preprocessing._data.MinMaxScaler ou sklearn.preprocessing._data.StandardScaler
            Objeto de scalling do sklearn, já ajustado para todos os dados

        Retorna:
        --------
        As saídas previstas já com a escala ajustada
        """
        
        if not (type(X_teste) is np.ndarray):
            raise TypeError("Os dados de entrada de teste devem ser um array do numpy!")

        if not ((type(scaler) is not None) and
                ((type(scaler) is sklearn.preprocessing._data.MinMaxScaler) or
                (type(scaler) is sklearn.preprocessing._data.StandardScaler))):
            raise TypeError("O scaler deve ser um MinMaxScaler ou StandardScaler!")

        modelo = self._modelo
        shape = self._shape

        # formatando os dados de entrada
        len_teste = X_teste.shape[0]
        n_samples = X_teste.shape[1]

        # formatando os dados de entrada para o many-to-one
        if (shape == 'many-to-one'):
            X_teste = np.reshape(X_teste,(len_teste, n_samples, 1))

        # formatando os dados de entrada para o one-to-one
        elif (shape == 'one-to-one'):
            X_teste = np.reshape(X_teste,(len_teste, 1, n_samples))

        y_pred = modelo.predict(X_teste)

        # caso a escala dos dados tiver sido alterada, desfaz ela
        if (scaler != None):
            y_pred = scaler.inverse_transform(y_pred)

        return y_pred
    
    def avaliar(self, X_treino, X_val, X_teste, y_treino,
                y_val, y_teste, n_repeticoes = 5, batch_size=10,
                early_stopping="ON", epochs=100,
                scaler=None):
        """
        Definição:
        ----------
        Função para treinar a rede e prever os dados n_repeticoes de vezes de forma a obter 
        uma média e um desvio padrão para o erro quadrático médio
        
        Ela deve ser executada antes do fit! Ou seja, executar após o construir_mlp() e o compilar()

        Por padrão, ela formata os dados em Many-to-One

        Se os dados foram ajustados por um scaler, passar ele como uma entrada
        
        Parâmetros:
        -----------
        X_treino: np.ndarray
            Conjunto de entradas para o treinamento
        X_val: np.ndarray
            Conjunto de entradas para a validação
        X_teste: np.ndarray
            Conjunto de entradas para os dados de teste            
        y_treino: np.ndarray
            Conjunto de saidas para o treinamento
        y_val: np.ndarray
            Conjunto de saídas para a validação
        y_teste: np.ndarray
            Conjunto de saídas para o teste      
        n_repeticoes: int
            Número de repetições a serem feitas            
        batch_size: int
            Tamanho do batch a ser utilizado durante o treinamento
        early_stopping: str
            Se deve "ON" ou não deve "OFF" utilizar early stopping
        epochs: int
            Número de épocas para o treinamento
        scaler: sklearn.preprocessing._data.MinMaxScaler ou sklearn.preprocessing._data.StandardScaler
            Objeto de scalling do sklearn, já ajustado para todos os dados

        Retorna:
        --------
        A média e desvio padrão do erro quadrático médio para essa rede neural,
        além de uma mensagem com essas informações
        """

        if not (type(n_repeticoes) is int):
            raise TypeError("O número de repetições deve ser um inteiro!")
        
        if not (type(X_treino) is np.ndarray):
            raise TypeError("Os dados de entrada de treino devem ser um array do numpy!")  
            
        if not (type(X_val) is np.ndarray):
            raise TypeError("Os dados de entrada de validação devem ser um array do numpy!")  

        if not (type(X_teste) is np.ndarray):
            raise TypeError("Os dados de entrada de teste devem ser um array do numpy!")            
            
        if not (type(y_treino) is np.ndarray):
            raise TypeError("Os dados de saída de treino devem ser um array do numpy!")  
            
        if not (type(y_val) is np.ndarray):
            raise TypeError("Os dados de saída de validação devem ser um array do numpy!")  

        if not (type(y_teste) is np.ndarray):
            raise TypeError("Os dados de saída de teste devem ser um array do numpy!")              
            
        if not (type(batch_size) is int):
            raise TypeError("O batch size deve ser um inteiro!")
            
        if not (type(early_stopping) is str):
            raise TypeError("O parâmetro de early stopping deve ser uma string!")

        if not ((early_stopping == 'ON') | (early_stopping == 'OFF')):
            raise ValueError("A string do early stopping deve ser um ON ou OFF")            
            
        if not (type(epochs) is int):
            raise TypeError("O número de épocas deve ser um int!")        
        
        if not ((type(scaler) is not None) and
                ((type(scaler) is sklearn.preprocessing._data.MinMaxScaler) or
                (type(scaler) is sklearn.preprocessing._data.StandardScaler))):
            raise TypeError("O scaler deve ser um MinMaxScaler ou StandardScaler!")

        shape = self._shape

        len_treino = X_treino.shape[0]
        len_val = X_val.shape[0]
        len_teste = X_teste.shape[0]
        n_samples = X_treino.shape[1]

        # formatando os dados de entrada para o many-to-one
        if (shape == 'many-to-one'):
            X_treino = np.reshape(X_treino,(len_treino, n_samples, 1))
            X_val = np.reshape(X_val,(len_val, n_samples, 1))
            X_teste = np.reshape(X_teste,(len_teste, n_samples, 1))

        # formatando os dados de entrada para o one-to-one
        elif (shape == 'one-to-one'):
            X_treino = np.reshape(X_treino,(len_treino, 1, n_samples))
            X_val = np.reshape(X_val,(len_val, 1, n_samples))
            X_teste = np.reshape(X_teste,(len_teste, 1, n_samples))

        if (scaler != None):
            y_teste = scaler.inverse_transform(y_teste)

        conjunto_mse = []
        
        if (early_stopping == 'ON'):
            early_stopping_fit = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss')
        else:
            early_stopping_fit = None
        
        for n in range(0, n_repeticoes):
            print("Testando para a repetição de número " + str(n+1))
            modelo = self._modelo
            
            modelo.fit(X_treino, y_treino, 
                   epochs=epochs, callbacks=early_stopping_fit,
                   validation_data=(X_val, y_val), batch_size=batch_size,
                   verbose=0)
            
            y_pred = modelo.predict(X_teste)

            # caso a escala dos dados tiver sido alterada, desfaz ela para medirmos o mse correto
            if (scaler != None):
                y_pred = scaler.inverse_transform(y_pred)

            mse = mean_squared_error(y_teste, y_pred)
            print("MSE para essa repetição: " + str(mse))
            conjunto_mse.append(mse)
        
        mse_med = statistics.mean(conjunto_mse)
        mse_dev = statistics.stdev(conjunto_mse)
        
        print("Média do erro quadrático médio: " + str(mse_med))
        print("Desvio padrão do erro quadrático médio: " + str(mse_dev) + "\n")
        
        return (mse_med, mse_dev)
    
    def salvar(self, nome_do_arquivo, h5="OFF"):
        """
        Definição:
        ----------
        Função para salvar o modelo num arquivo .h5 ou no padrão do Tensor Flow
        
        Parâmetros:
        -----------
        nome_do_arquivo: str
            Nome do arquivo a ser salvo, podendo incluir o caminho
        h5: str
            Se deve "ON" ou não deve "OFF" salvar no formato .h5
            
        Retorna:
        --------
        Uma mensagem confirmando o salvamento
        """
        
        if not (type(nome_do_arquivo) is str):
            raise TypeError("O parâmetro de nome do arquivo deve ser uma string!")        

        if not (type(h5) is str):
            raise TypeError("O parâmetro de h5 deve ser uma string!")

        if not ((h5 == 'ON') | (h5 == 'OFF')):
            raise ValueError("A string do h5 deve ser um ON ou OFF")             
        
        modelo = self._modelo
        
        if (h5 == "ON"):
            modelo.save(nome_do_arquivo, include_optimizer=True, save_format="h5")
        else:
            modelo.save(nome_do_arquivo, include_optimizer=True)
        
        return print("O modelo foi salvo!")
    
    def carregar(self, nome_do_arquivo):
        """
        Definição:
        ----------
        Função para carregar o modelo através de um arquivo .h5 ou padrão do Tensor Flow
        
        Parâmetros:
        -----------
        nome_do_arquivo: str
            Nome do arquivo a ser salvo, podendo incluir o caminho
            Deve incluir o formato!
            
        Retorna:
        --------
        Uma mensagem confirmando o carregamento
        """
        
        if not (type(nome_do_arquivo) is str):
            raise TypeError("O parâmetro de nome do arquivo deve ser uma string!")        
        
        modelo = keras.models.load_model(nome_do_arquivo)
        self._modelo = modelo
        
        return print("O modelo foi carregado!")    