# pibic2020.models.esn_model.py

import numpy as np
from numpy.linalg import pinv, eigvals, inv

class ModeloESN:
    
    def __init__(self, n_neurons, leaky_values, spectral_radius, scale):
        """
        Descrição:
        ----------
        Construtor da classe 'ModeloESN'

        Parâmetros:
        -----------
        num_neuronios: int
            Número de neurônios por reservatório
        leaky_values: np.ndarray
            Lista com os coeficientes de vazamento
        spectral_radius: np.ndarray
            Lista com os raios espectrais
        scale: int ou float
            Valor absoluto máximo aceito para os pesos de entrada

        Retorna:
        --------
        Nada
        """

        if not (type(n_neurons) is int):
            raise TypeError("O número de neurônios deve ser um inteiro!")

        if not (type(leaky_values) is np.ndarray):
            raise TypeError("A lista com os coeficientes de vazamento deve ser um array do numpy!")   

        if not (type(spectral_radius) is np.ndarray):
            raise TypeError("A lista com os raios espectrais deve ser um array do numpy!")                 

        self._n_neurons = n_neurons
        self._leaky_values = leaky_values
        self._spectral_radius = spectral_radius
        self._win_max = scale

        # inicializa alguns parâmetros padrões da rede
        self._n_reserv = 1
        self._W_in = []
        self._W_reserv = []
        self._state_reserv = []
        self._reserv_vector = []
        self._W_out = []
        pass

    def treinar(self, X_treino, y_treino, n_transitory_samples):
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
        n_transitory_samples: int
            Número de amostras para o transitório (inicializar o vetor de estado)

        Retorna:
        --------
        Nada
        """

        # ajusta o formato das entradas
        X_treino = X_treino.T
        y_treino = y_treino.T

        # extrai o número de atributos de entrada e o número de padrões do conjunto de treinamento (com transitório)
        K, n_samples = X_treino.shape
        
        # número de padrões efetivos do conjunto de treinamento
        n_effective_samples = n_samples - n_transitory_samples
        
        # inicializa a matriz que contém a concatenação dos estados dos reservatórios
        self._reserv_vector = np.zeros((self._n_neurons*self._n_reserv, n_effective_samples))
        
        for l in range(0, self._n_reserv):
            
            # inicializa a matriz com os estados do l-ésimo reservatório
            layer_state = np.zeros((self._n_neurons, n_samples + 1))
            
            # número de entradas do reservatório l
            if (l == 0):
                n_layer_inputs = K
                layer_input = X_treino
            else:
                n_layer_inputs = self._n_neurons
                
                # a entrada da l-ésima camada é o vetor de saída da (l-1)-ésima camada (excluindo o estado inicial)
                layer_input = np.delete(self._state_reserv[l-1], 0, 1)
            
            # matriz de pesos de entrada (W_layer_in) do reservatório l: n_neurons x n_layer_inputs
            W_layer_in = 2*self._win_max*np.random.rand(self._n_neurons, n_layer_inputs) - self._win_max
            self._W_in.append(W_layer_in)
            
            # matriz de pesos recorrentes do reservatório l
            W_layer = 2*np.random.rand(self._n_neurons, self._n_neurons) - 1
            W_spectral = (1 - self._leaky_values[l])*np.eye(self._n_neurons) + self._leaky_values[l]*W_layer
            max_eigenvalue = max(abs(eigvals(W_spectral)))
            Ws = (self._spectral_radius[l]/max_eigenvalue)*W_spectral
            W_layer = (1/self._leaky_values[l])*(Ws - (1 - self._leaky_values[l])*np.eye(self._n_neurons))
            self._W_reserv.append(W_layer)
            
            # computa o estado do reservatório l para todos os instantes do conjunto de treinamento
            for i in range(0, n_samples):
                
                layer_state[:, i + 1] = (1-self._leaky_values[l])*layer_state[:, i] + self._leaky_values[l]*np.tanh(np.matmul(self._W_in[l], layer_input[:, i]) + np.matmul(self._W_reserv[l], layer_state[:, i]))
                    
            self._state_reserv.append(layer_state)
            
        for l in range(0, self._n_reserv):           
            # elimina a primeira coluna (estado inicial com zeros) e os primeiros n_transitory_samples estados (transitório)
            # concatena a matriz de estados do reservatório l ao repositório completo
            self._reserv_vector[l*self._n_neurons:(l + 1)*self._n_neurons, :] = self._state_reserv[l][:, n_transitory_samples + 1:]
        
        # Agora, basta computar a pseudo-inversa da matriz reserv_vector para determinar os pesos da camada de saída         
        self._W_out = np.matmul(pinv(self._reserv_vector.T), y_treino[0, n_transitory_samples:].T)
        self._W_out = self._W_out.T
        pass

    def predicao(self, X_teste, n_transitory_samples):
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

        if not (type(n_transitory_samples) is int):
            raise TypeError("O número de amostras para o transitório deve ser um inteiro!") 
        
        # ajusta o formato da entrada
        X_teste = X_teste.T

        # extrai o número de padrões do conjunto de teste
        n_test_samples = X_teste.shape[1]
        
        # inicializa a matriz com os estados concatenados (partimos do último estado observado no treinamento)
        reserv_vector_test = np.zeros((self._n_neurons*self._n_reserv, n_test_samples - n_transitory_samples))
        
        # inicializa a lista que vai guardar as matrizes com os estados de cada reservatório (para todos os instantes de teste)
        state_reserv_test = []
        
        for l in range(0, self._n_reserv):
            
            # inicializa a matriz com os estados do l-ésimo reservatório (o estado inicial equivale ao último do treinamento)
            layer_state = np.zeros((self._n_neurons, n_test_samples + 1))
            layer_state[:, 0] = self._state_reserv[l][:, -1]
            
            if (l == 0):
                layer_input = X_teste
            else:
                # a entrada da l-ésima camada é o vetor de saída da (l-1)-ésima camada (excluindo o estado inicial)
                layer_input = state_reserv_test[l - 1]
            
            # computa o estado do reservatório l para todos os instantes do conjunto de teste
            for i in range(0, n_test_samples):
                layer_state[:, i + 1] = (1 - self._leaky_values[l])*layer_state[:, i] + self._leaky_values[l]*np.tanh(np.matmul(self._W_in[l], layer_input[:, i]) + np.matmul(self._W_reserv[l], layer_state[:, i]))
                    
            # elimina a primeira coluna (estado inicial com zeros)
            state_reserv_test.append(np.delete(layer_state, 0, 1))
        
        # elimina os primeiros n_transitory_samples estados de todas as camadas
        for l in range(0, self._n_reserv):
            reserv_vector_test[l*self._n_neurons:(l + 1)*self._n_neurons, :] = np.delete(state_reserv_test[l], np.arange(n_transitory_samples), 1)
            
        # gera as saídas para o conjunto de teste
        y_predicao = np.matmul(self._W_out, reserv_vector_test).T

        return y_predicao

    def _resetar_rede(self):
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