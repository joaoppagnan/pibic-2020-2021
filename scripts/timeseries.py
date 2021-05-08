# timeseries.py

import numpy as np

class SerieTemporal:

    def __init__(self, dados, K, L):
        """
        Descrição:
        ----------
        Construtor da classe 'SerieTemporal'

        Parâmetros:
        -----------
        dados: np.ndarray
            Conjunto de valores da série temporal
        K: int
            O número de entradas utilizado para a predição
        L: int
            O passo de predição     
            
        Retorna:
        --------
        Nada
        """

        if not type(dados) is np.ndarray:
            raise TypeError("Os dados devem ser um array do numpy!")        

        if (K <= 0):
            raise ValueError("O hiperparâmetro 'K' deve ser um inteiro positivo!")

        if (L > len(dados)):
            raise ValueError("L deve ser menor que o número de dados temporais!")

        self.__dados = dados
        self.__K = K
        self.__L = L
        self._matriz_entrada = np.array([])
        self._matriz_saida = np.array([])
        pass

    def _criar_vetores(self, indice):
        """
        Descrição:
        ----------
        Função interna para criar os vetores de entrada e saída para as etapas de treinamento e teste para uma série temporal
    
        Parâmetros:
        -----------
        indice: int
            A posicao a partir da qual se deseja prever algum valor
            
        Retorna:
        --------
        Os vetores de entrada e saída para o índice escolhido no formato np.ndarray
        """

        K, L = self.__K, self.__L
        dados = self.__dados
    
        # checa se a partir da posição atual podemos criar um vetor de amostras dado um K
        if ((indice + 1) < (K - 1)):
            raise ValueError("(indice + 1) = "+str(indice + 1)+" deve ser maior ou igual a (K - 1) = "+str(K - 1)+" !")
        
        # checa se o valor que queremos prever (que vai ser armazenado na matriz de saida), está dentro da série temporal
        if ((indice+L) > (len(dados)-1)):
            raise ValueError("O passo de predição (L = "+str(L)+") somado com o índice atual (indice = "+str(indice)+") não deve estourar o número de dados na série temporal!")
    
        vetor_entrada = np.array(dados[(indice-(K-1)):(indice+1)])
        vetor_saida = np.array(dados[indice+L])

        return vetor_entrada, vetor_saida    

    def dividir_treino_teste(self, tam_teste):
        """
        Descrição:
        ----------
        Função para selecionar os tam_teste*len(matriz_entrada) últimos dados das matrizes para o teste

        Parâmetros:
        -----------
        tam_teste: float
            Proporção de dados que iremos separar para o teste. Deve ser entre 0.0 e 1.0
            
        Retorna:
        --------
        O conjunto de teste e treinamento para a proporção solicitada no formato np.ndarray
        """
        
        if ((tam_teste < 0.0) | (tam_teste > 1.0)):
            raise ValueError("A proporção dos dados de teste deve ser entre 0.0 e 1.0!")
            
        matriz_entrada = self._matriz_entrada
        matriz_saida = self._matriz_saida
            
        tam_treino = 1.0 - tam_teste
        n_dados = len(matriz_saida)

        X_treino = np.array(matriz_entrada[:int(tam_treino*n_dados)])
        X_teste = np.array(matriz_entrada[int(n_dados*(1 - tam_teste)):])
        y_treino = np.array(matriz_saida[:int(tam_treino*n_dados)])
        y_teste = np.array(matriz_saida[int(n_dados*(1 - tam_teste)):])
        
        return (X_treino, X_teste,
                y_treino, y_teste)

    def criar_matrizes(self):
        """
        Descrição:
        ----------
        Função para criar as matrizes com os vetores de entrada e saída para as etapas de treinamento e teste
    
        Parâmetros:
        -----------
        Nenhum
        
        Retorna:
        --------
        Nada
        """
    
        K, L = self.__K, self.__L
        num_dados = len(self.__dados)
    
        for indice in range((K-1), (num_dados-L)):
            vetor_entrada, vetor_saida = self._criar_vetores(indice)
        
            if (len(self._matriz_entrada) == 0):
                self._matriz_entrada = vetor_entrada
            else:
                self._matriz_entrada = np.vstack((self._matriz_entrada, vetor_entrada))
                
            if (len(self._matriz_saida) == 0):
                self._matriz_saida = np.array([vetor_saida])
            else:
                self._matriz_saida = np.vstack((self._matriz_saida, vetor_saida))
        pass