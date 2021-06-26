# pibic2020.models.esn_model.py

import numpy as np
from numpy.linalg import pinv, eigvals, inv

class DeepESN:
    
    """ Inicialização de um objeto da classe DeepESN 
    
    Parâmetros: num_neuronios = número de neurônios por reservatório
                num_camadas =  número de camadas da rede (num_camadas >= 1)
                leaky_values = lista com os coeficientes de vazamento de cada camada
                spectral_radii = lista com os raios espectrais de cada camada
                scale_in = valor absoluto máximo aceito para os pesos de entrada
    """
    def __init__(self,num_neuronios,num_camadas,leaky_values,spectral_radii,scale_in):
        self.N = num_neuronios
        self.Nl = num_camadas
        self.a = leaky_values
        self.rho = spectral_radii
        self.win_max = scale_in
        
    """ Método fit: 
          * constrói as matrizes aleatórias do reservatório
          * computa a matriz com os estados da rede
          * obtém a matriz de pesos de saída através da solução de quadrados mínimos (pseudo-inversa)

          Parâmetros: u - matriz com os atributos de entrada de todos os padrões de treinamento (K x T + Tr)
                      d - matriz com as saídas desejadas (L x T)
                      Tr - número de amostras para o transitório (inicializar o vetor de estado)
    """

    def fit(self,u,d,Tr):

        u=u.T
        d=d.T

        #extrai o número de atributos de entrada e o número de padrões do conjunto de treinamento (com transitório)
        self.K,self.Tt = u.shape
        
        #número de padrões efetivos do conjunto de treinamento
        self.T = self.Tt - Tr
        
        #extrai o número de saídas da rede
        self.L = d.shape[0]
        
        """ Para cada camada l (de 1 a Nl), devemos determinar (1) a matriz de pesos de entrada (aleatória),
        (2) a matriz de pesos recorrentes, respeitando a condição de estados de eco, e (3) o vetor de estados (para
        todos os instantes de tempo) 
        Por simplicidade, vamos guardar estas informações em filas de comprimento Nl 
        """
        
        #inicializa as listas que vão guardar as matrizes de pesos Win, W
        self.Win = []
        self.W = []
        #inicializa a lista que vai guardar as matrizes com os estados de cada reservatório (para todos os instantes)
        self.X = []
        #inicializa a matriz que contém a concatenação dos estados dos reservatórios
        self.Xvec = np.zeros((self.N*self.Nl,self.T))
        
        for l in range(0,self.Nl):
            
            #inicializa a matriz com os estados do l-ésimo reservatório
            x_l = np.zeros((self.N,self.Tt+1))
            
            #número de entradas do reservatório l
            if (l == 0):
                Ne_l = self.K
                U = u
            else:
                Ne_l = self.N
                #a entrada da l-ésima camada é o vetor de saída da (l-1)-ésima camada (excluindo o estado inicial)
                U = np.delete(self.X[l-1],0,1)
            
            #matriz de pesos de entrada (Win) do reservatório l: N x Ne
            Win_l = 2*self.win_max*np.random.rand(self.N,Ne_l) - self.win_max
            self.Win.append(Win_l)
            
            #matriz de pesos recorrentes do reservatório l
        
            W_l = 2*np.random.rand(self.N,self.N) - 1
            Ws = (1-self.a[l])*np.eye(self.N) + self.a[l]*W_l
            max_v = max(abs(eigvals(Ws)))
            Ws = (self.rho[l]/max_v)*Ws
            W_l = (1/self.a[l])*(Ws - (1-self.a[l])*np.eye(self.N))
            self.W.append(W_l)
            
            #computa o estado do reservatório l para todos os instantes do conjunto de treinamento
            for i in range(0,self.Tt):
                
                x_l[:,i+1] = (1-self.a[l])*x_l[:,i] + self.a[l]*np.tanh(np.matmul(self.Win[l],U[:,i]) + np.matmul(self.W[l],x_l[:,i]))
                    
            self.X.append(x_l)
            
        for l in range(0,self.Nl):
            
            aux = self.X[l]
            """ elimina a primeira coluna (estado inicial com zeros) e os primeiros Tr estados (transitório)
                concatena a matriz de estados do reservatório l ao repositório completo """
            self.Xvec[l*self.N:(l+1)*self.N,:] = aux[:,Tr+1:]
        
        """ Agora, basta computar a pseudo-inversa da matriz Xvec para determinar os pesos da camada de saída """        
        self.Wout = np.matmul(pinv(self.Xvec.T),d[0,Tr:].T)
        self.Wout = self.Wout.T
        pass


    """ Método predict 
    
        Obtém as saídas da DeepESN para o conjunto de teste; as matrizes de pesos já estão prontas
    
    """
    def predict(self,ut,dt,Tr):
        
        ut=ut.T
        dt=dt.T

        #extrai o número de padrões do conjunto de teste
        self.nt = ut.shape[1]
        
        #inicializa a matriz com os estados concatenados (partimos do último estado observado no treinamento)
        self.Xvec_teste = np.zeros((self.N*self.Nl,self.nt-Tr))
        
        #inicializa a lista que vai guardar as matrizes com os estados de cada reservatório (para todos os instantes de teste)
        self.X_teste = []
        
        for l in range(0,self.Nl):
            
            #inicializa a matriz com os estados do l-ésimo reservatório (o estado inicial equivale ao último do treinamento)
            x_l = np.zeros((self.N,self.nt+1))
            Xl_aux = self.X[l]
            x_l[:,0] = Xl_aux[:,-1]
            
            if (l == 0):
                U = ut
            else:
                #a entrada da l-ésima camada é o vetor de saída da (l-1)-ésima camada (excluindo o estado inicial)
                U = self.X_teste[l-1]
            
            #computa o estado do reservatório l para todos os instantes do conjunto de teste
            for i in range(0,self.nt):
                
                x_l[:,i+1] = (1-self.a[l])*x_l[:,i] + self.a[l]*np.tanh(np.matmul(self.Win[l],U[:,i]) + np.matmul(self.W[l],x_l[:,i]))
                    
            #elimina a primeira coluna (estado inicial com zeros) 
            xt = np.delete(x_l, 0,1)
            self.X_teste.append(xt)
        
        for l in range(0,self.Nl):
            #elimina os primeiros Tr estados (transitório)
            Xaux = self.X_teste[l]
            xt = np.delete(Xaux,np.arange(Tr),1)
            self.Xvec_teste[l*self.N:(l+1)*self.N,:] = xt
            
        #gera as saídas para o conjunto de teste
        yteste = np.matmul(self.Wout,self.Xvec_teste)
        yteste = yteste.T

        return yteste