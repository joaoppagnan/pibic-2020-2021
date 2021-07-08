# pibic2020.models.esn_model.py

from typing import Type
import numpy as np
from numpy.linalg import pinv, eigvals
from sklearn.base import BaseEstimator

import statistics
from sklearn.metrics import mean_squared_error

class ModeloESN(BaseEstimator):
    
    def __init__(self, n_neurons=30, spectral_radius=[1], win_max=1, leaky_values=[0.9], percent_transitory_samples=0.0):
        """
        Descrição:
        ----------
        Construtor da classe 'ModeloESN'

        Parâmetros:
        -----------
        num_neuronios: int
            Número de neurônios por reservatório
        leaky_values: list
            Lista com os coeficientes de vazamento
        spectral_radius: list
            Valor com o raio espectral
        scale: int ou float
            Valor absoluto máximo aceito para os pesos de entrada
        percent_transitory_samples = float
            Fração de valores de entrada que estarão no conjunto do transitório

        Retorna:
        --------
        Nada
        """

        if not ((type(n_neurons) is int) or (n_neurons is None)):
            raise TypeError("O número de neurônios deve ser um inteiro!")

        if not (type(leaky_values) is list):
            raise TypeError("A lista com os coeficientes de vazamento deve ser uma lista!")   

        if not (type(spectral_radius) is list):
            raise TypeError("O raio espectral deve ser uma lista!")                 

        if not (type(percent_transitory_samples) is float):
            raise TypeError("A fração de amostras do transitório deve ser um float!")
            
        self.n_neurons = n_neurons
        self.leaky_values = leaky_values
        self.spectral_radius = spectral_radius
        self.win_max = win_max
        self.percent_transitory_samples = percent_transitory_samples

        # inicializa alguns parâmetros padrões da rede
        self._n_reserv = 1
        self._W_in = []
        self._W_reserv = []
        self._state_reserv = []
        self._reserv_vector = []
        self._W_out = []
        pass

    def fit(self, X_treino, y_treino):
        """
        Descrição:
        ----------
        Constrói as matrizes aleatórias do reservatório, computa a matriz com os estados da rede
        e obtém a matriz de pesos de saída através da solução de quadrados mínimos (pseudo-inversa)

        Parâmetros:
        -----------
        X_treino: np.ndarray
            Conjunto de entradas para o treinamento
        y_treino: np.ndarray
            Conjunto de saidas para o treinamento

        Retorna:
        --------
        Nada
        """

        if not (type(X_treino) is np.ndarray):
            raise TypeError("Os dados de entrada de treino devem ser um array do numpy!")

        if not (type(y_treino) is np.ndarray):
            raise TypeError("Os dados de saída de treino devem ser um array do numpy!")    

        # calcula o numero de amostras de transitorio
        n_transitory_samples = int(self.percent_transitory_samples*len(y_treino))

        # ajusta o formato das entradas
        X_treino = X_treino.T
        y_treino = y_treino.T

        # extrai o número de atributos de entrada e o número de padrões do conjunto de treinamento (com transitório)
        K, n_samples = X_treino.shape
        
        # número de padrões efetivos do conjunto de treinamento
        n_effective_samples = n_samples - n_transitory_samples
        
        # inicializa a matriz que contém a concatenação dos estados dos reservatórios
        self._reserv_vector = np.zeros((self.n_neurons*self._n_reserv, n_effective_samples))
        
        for l in range(0, self._n_reserv):
            
            # inicializa a matriz com os estados do l-ésimo reservatório
            layer_state = np.zeros((self.n_neurons, n_samples + 1))
            
            # número de entradas do reservatório l
            if (l == 0):
                n_layer_inputs = K
                layer_input = X_treino
            else:
                n_layer_inputs = self.n_neurons
                
                # a entrada da l-ésima camada é o vetor de saída da (l-1)-ésima camada (excluindo o estado inicial)
                layer_input = np.delete(self._state_reserv[l-1], 0, 1)
            
            # matriz de pesos de entrada (W_layer_in) do reservatório l: n_neurons x n_layer_inputs
            W_layer_in = 2*self.win_max*np.random.rand(self.n_neurons, n_layer_inputs) - self.win_max
            self._W_in.append(W_layer_in)
            
            # matriz de pesos recorrentes do reservatório l
            W_layer = 2*np.random.rand(self.n_neurons, self.n_neurons) - 1
            W_spectral = (1 - self.leaky_values[l])*np.eye(self.n_neurons) + self.leaky_values[l]*W_layer
            max_eigenvalue = max(abs(eigvals(W_spectral)))
            Ws = (self.spectral_radius[l]/max_eigenvalue)*W_spectral
            W_layer = (1/self.leaky_values[l])*(Ws - (1 - self.leaky_values[l])*np.eye(self.n_neurons))
            self._W_reserv.append(W_layer)
            
            # computa o estado do reservatório l para todos os instantes do conjunto de treinamento
            for i in range(0, n_samples):
                
                layer_state[:, i + 1] = (1-self.leaky_values[l])*layer_state[:, i] + self.leaky_values[l]*np.tanh(np.matmul(self._W_in[l], layer_input[:, i]) + np.matmul(self._W_reserv[l], layer_state[:, i]))
                    
            self._state_reserv.append(layer_state)
            
        for l in range(0, self._n_reserv):           
            # elimina a primeira coluna (estado inicial com zeros) e os primeiros n_transitory_samples estados (transitório)
            # concatena a matriz de estados do reservatório l ao repositório completo
            self._reserv_vector[l*self.n_neurons:(l + 1)*self.n_neurons, :] = self._state_reserv[l][:, n_transitory_samples + 1:]
        
        # Agora, basta computar a pseudo-inversa da matriz reserv_vector para determinar os pesos da camada de saída         
        self._W_out = np.matmul(pinv(self._reserv_vector.T), y_treino[0, n_transitory_samples:].T)
        self._W_out = self._W_out.T
        return self

    def predict(self, X_teste):
        """
        Descrição:
        ----------
        Obtém as saídas da ESN para o conjunto de teste

        Parâmetros:
        -----------
        X_teste: np.ndarray
            Conjunto de entradas para os dados de teste
        n_transitory_samples: int
            número de amostras para o transitório (inicializar o vetor de estado)

        Retorna:
        --------
        As saídas previstas
        """

        if not (type(X_teste) is np.ndarray):
            raise TypeError("Os dados de entrada de teste devem ser um array do numpy!")
        
        # calcula o numero de amostras de transitorio
        n_transitory_samples = int(self.percent_transitory_samples*len(X_teste))

        # ajusta o formato da entrada
        X_teste = X_teste.T

        # extrai o número de padrões do conjunto de teste
        n_test_samples = X_teste.shape[1]
        
        # inicializa a matriz com os estados concatenados (partimos do último estado observado no treinamento)
        reserv_vector_test = np.zeros((self.n_neurons*self._n_reserv, n_test_samples - n_transitory_samples))
        
        # inicializa a lista que vai guardar as matrizes com os estados de cada reservatório (para todos os instantes de teste)
        state_reserv_test = []
        
        for l in range(0, self._n_reserv):
            
            # inicializa a matriz com os estados do l-ésimo reservatório (o estado inicial equivale ao último do treinamento)
            layer_state = np.zeros((self.n_neurons, n_test_samples + 1))
            layer_state[:, 0] = self._state_reserv[l][:, -1]
            
            if (l == 0):
                layer_input = X_teste
            else:
                # a entrada da l-ésima camada é o vetor de saída da (l-1)-ésima camada (excluindo o estado inicial)
                layer_input = state_reserv_test[l - 1]
            
            # computa o estado do reservatório l para todos os instantes do conjunto de teste
            for i in range(0, n_test_samples):
                layer_state[:, i + 1] = (1 - self.leaky_values[l])*layer_state[:, i] + self.leaky_values[l]*np.tanh(np.matmul(self._W_in[l], layer_input[:, i]) + np.matmul(self._W_reserv[l], layer_state[:, i]))
                    
            # elimina a primeira coluna (estado inicial com zeros)
            state_reserv_test.append(np.delete(layer_state, 0, 1))
        
        # elimina os primeiros n_transitory_samples estados de todas as camadas
        for l in range(0, self._n_reserv):
            reserv_vector_test[l*self.n_neurons:(l + 1)*self.n_neurons, :] = np.delete(state_reserv_test[l], np.arange(n_transitory_samples), 1)
            
        # gera as saídas para o conjunto de teste
        y_predicao = np.matmul(self._W_out, reserv_vector_test).T

        return y_predicao

    def avaliar(self, X_treino, X_teste, y_treino, y_teste, 
                n_repeticoes = 5, verbose=0):
        """
        Definição:
        ----------
        Função para treinar a rede e prever os dados n_repeticoes de vezes de forma a obter 
        uma média e um desvio padrão para o erro quadrático médio
        
        Ela deve ser executada antes do fit!
        
        Parâmetros:
        -----------
        X_treino: np.ndarray
            Conjunto de entradas para o treinamento
        X_teste: np.ndarray
            Conjunto de entradas para os dados de teste            
        y_treino: np.ndarray
            Conjunto de saidas para o treinamento
        y_teste: np.ndarray
            Conjunto de saídas para o teste      
        n_repeticoes: int
            Número de repetições a serem feitas
        verbose: int
            Se vai retornar mensagens ao longo do processo (0 ou 1)

        Retorna:
        --------
        A média e desvio padrão do erro quadrático médio para essa rede neural,
        além de uma mensagem com essas informações
        """

        if not (type(n_repeticoes) is int):
            raise TypeError("O número de repetições deve ser um inteiro!")
        
        if not (type(X_treino) is np.ndarray):
            raise TypeError("Os dados de entrada de treino devem ser um array do numpy!")  

        if not (type(X_teste) is np.ndarray):
            raise TypeError("Os dados de entrada de teste devem ser um array do numpy!")            
            
        if not (type(y_treino) is np.ndarray):
            raise TypeError("Os dados de saída de treino devem ser um array do numpy!")  

        if not (type(y_teste) is np.ndarray):
            raise TypeError("Os dados de saída de teste devem ser um array do numpy!")

        if not ((type(verbose) is int) and
                ((verbose == 0) or
                 (verbose == 1)
                 (verbose == 2))):
            raise ValueError("O valor de verbose deve ser um int igual a 0, 1 ou 2!")  

        conjunto_mse = []
        
        for n in range(0, n_repeticoes):
            if (verbose == 2):
                print("Testando para a repetição de número " + str(n+1))
            modelo = self
            
            modelo.fit(X_treino, y_treino)
            
            y_pred = modelo.predict(X_teste)

            mse = mean_squared_error(y_teste, y_pred)
            if (verbose == 2):
                print("MSE para essa repetição: " + str(mse))
            conjunto_mse.append(mse)
            modelo._reset()
        
        mse_med = statistics.mean(conjunto_mse)
        mse_dev = statistics.stdev(conjunto_mse)
        
        if (verbose == 1):
            print("Média do erro quadrático médio: " + str(mse_med))
            print("Desvio padrão do erro quadrático médio: " + str(mse_dev) + "\n")
        
        return mse_med, mse_dev        

    def _reset(self):
        """
        Descrição:
        ----------
        Método interno para resetar os parâmetros da rede estimados com o treinar().
        Utilizado no método avaliar()

        Parâmetros:
        -----------
        Nenhum

        Retorna:
        --------
        Nada
        """

        self._W_in = []
        self._W_reserv = []
        self._state_reserv = []
        self._reserv_vector = []
        self._W_out = []
        pass