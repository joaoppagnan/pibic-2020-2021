# pibic2020.models.esn_model.py

import numpy as np
from numpy.linalg import pinv, eigvals, inv

class ModeloESN():

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
        self._a = leaky_values
        self.rho = spectral_radius
        self.win_max = scale
        self.Nl = 1

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

        if not (type(X_treino) is np.ndarray):
            raise TypeError("Os dados de entrada de treino devem ser um array do numpy!")  
            
        if not (type(y_treino) is np.ndarray):
            raise TypeError("Os dados de saída de treino devem ser um array do numpy!")  

        if not (type(n_transitory_samples) is int):
            raise TypeError("O número de amostras para o transitório deve ser um inteiro!")    

        # extrai o número de atributos de entrada e o número de padrões do conjunto de treinamento (com transitório)
        self.Tt, self.K = X_treino.shape
        
        # número de padrões efetivos do conjunto de treinamento
        self.T = self.Tt - n_transitory_samples
        
        # extrai o número de saídas da rede
        self.L = y_treino.shape[0]
        
        # Para cada camada l (de 1 a Nl), devemos determinar (1) a matriz de pesos de entrada (aleatória),
        # (2) a matriz de pesos recorrentes, respeitando a condição de estados de eco, e (3) o vetor de estados (para
        # todos os instantes de tempo) 
        # Por simplicidade, vamos guardar estas informações em filas de comprimento Nl 

        
        # inicializa as listas que vão guardar as matrizes de pesos Win, W
        self.Win = []
        self.W = []

        # inicializa a lista que vai guardar as matrizes com os estados de cada reservatório (para todos os instantes)
        self.X = []

        # inicializa a matriz que contém a concatenação dos estados dos reservatórios
        self.Xvec = np.zeros((self.T, self._n_neurons*self.Nl))
        
        for l in range(0, self.Nl):
            
            # inicializa a matriz com os estados do l-ésimo reservatório
            x_l = np.zeros((self.Tt + 1, self._n_neurons))
            
            # número de entradas do reservatório l
            if (l == 0):
                Ne_l = self.K
                U = X_treino
            else:
                Ne_l = self._n_neurons
                # a entrada da l-ésima camada é o vetor de saída da (l-1)-ésima camada (excluindo o estado inicial)
                U = np.delete(self.X[l-1], 0, 1)
            
            # matriz de pesos de entrada (Win) do reservatório l: N x Ne
            Win_l = 2*self.win_max*np.random.rand(self._n_neurons, Ne_l) - self.win_max
            self.Win.append(Win_l)
            
            # matriz de pesos recorrentes do reservatório l
            W_l = 2*np.random.rand(self._n_neurons, self._n_neurons) - 1
            Ws = (1-self._a[l])*np.eye(self._n_neurons) + self._a[l]*W_l
            max_v = max(abs(eigvals(Ws)))
            Ws = (self.rho[l]/max_v)*Ws
            W_l = (1/self._a[l])*(Ws - (1-self._a[l])*np.eye(self._n_neurons))
            self.W.append(W_l)
            
            # computa o estado do reservatório l para todos os instantes do conjunto de treinamento
            for i in range(0,self.Tt):
                x_l[i+1, :] = (1-self._a[l])*x_l[i, :] + self._a[l]*np.tanh(np.matmul(self.Win[l],U[i, :]) + np.matmul(self.W[l],x_l[i, :]))
                    
            self.X.append(x_l)
            
        for l in range(0,self.Nl):
            
            aux = self.X[l]
            # elimina a primeira coluna (estado inicial com zeros) e os primeiros Tr estados (transitório)
            #    concatena a matriz de estados do reservatório l ao repositório completo
            self.Xvec[:, l*self._n_neurons:(l+1)*self._n_neurons] = aux[n_transitory_samples+1:, :]
        
        # Agora, basta computar a pseudo-inversa da matriz Xvec para determinar os pesos da camada de saída     
        self.Wout = np.matmul(pinv(self.Xvec), y_treino[n_transitory_samples:,])

        # saídas da rede para o conjunto de treinamento
        self.y_treino = np.matmul(self.Xvec, self.Wout)
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

        # extrai o número de padrões do conjunto de teste
        self.nt = X_teste.shape[0]
        
        #inicializa a matriz com os estados concatenados (partimos do último estado observado no treinamento)
        self.Xvec_teste = np.zeros((self.nt-n_transitory_samples, self._n_neurons*self.Nl))
        
        #inicializa a lista que vai guardar as matrizes com os estados de cada reservatório (para todos os instantes de teste)
        self.X_teste = []
        
        for l in range(0,self.Nl):
            
            # inicializa a matriz com os estados do l-ésimo reservatório (o estado inicial equivale ao último do treinamento)
            x_l = np.zeros((self.nt+1, self._n_neurons))
            Xl_aux = self.X[l]
            x_l[:, 0] = Xl_aux[:, -1]
            
            if (l == 0):
                U = X_teste
            else:
                # a entrada da l-ésima camada é o vetor de saída da (l-1)-ésima camada (excluindo o estado inicial)
                U = self.X_teste[l-1]
            
            # computa o estado do reservatório l para todos os instantes do conjunto de teste
            for i in range(0,self.nt):
                
                x_l[i+1, :] = (1-self._a[l])*x_l[i, :] + self._a[l]*np.tanh(np.matmul(self.Win[l],U[i, :]) + np.matmul(self.W[l],x_l[i, :]))
                    
            # elimina a primeira coluna (estado inicial com zeros) 
            xt = np.delete(x_l, 1, 1)
            self.X_teste.append(xt)
        
        for l in range(0, self.Nl):

            # elimina os primeiros n_transitory_samples estados (transitório)
            Xaux = self.X_teste[l]
            xt = np.delete(Xaux, np.arange(n_transitory_samples), 1)
            self.Xvec_teste[: l*self._n_neurons:, (l+1)*self._n_neurons] = xt
            
        # gera as saídas para o conjunto de teste
        y_predicao = np.matmul(self.Xvec_teste, self.Wout)
        return y_predicao