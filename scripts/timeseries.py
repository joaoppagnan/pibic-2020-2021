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
        dados: np.array
            Conjunto de valores da série temporal
        K: int
            O número de entradas utilizado para a predição
        L: int
            O passo de predição     
        """

        if (K <= 0):
            raise ValueError("O hiperparâmetro 'K' deve ser um inteiro positivo!")

        if not type(dados) is np.ndarray:
            raise TypeError("Os dados devem ser um array do numpy!")

        self._dados = dados
        self._K = K
        self._L = L
        self._matriz_entrada = np.array([])
        self._matriz_saida = np.array([])



    def dividir_treino_teste(self, tam_teste):
        """
        Descrição:
        ----------
        Função para selecionar os 'n' primeiros dados das matrizes para o treinamento.

        Parâmetros:
        -----------
        tam_teste: float
            Proporção de dados que iremos separar para o teste. Deve ser entre 0.0 e 1.0.
        """
        
        if ((tam_teste < 0.0) | (tam_teste > 1.0)):
            raise ValueError("A proporção dos dados de teste deve ser entre 0.0 e 1.0!")
            
        tam_treino = 1.0 - tam_teste
        n_dados = len(self._matriz_saida)

        K = self._K
        return (self._matriz_entrada[:int(tam_treino*n_dados)], self._matriz_entrada[int(n_dados*(1 - tam_teste)):],
                self._matriz_saida[:int(tam_treino*n_dados)], self._matriz_saida[int(n_dados*(1 - tam_teste)):])

    def conjunto_teste(self, n):
        """
        Descrição:
        ----------
        Função para selecionar os 'n' últimos dados das matrizes para o teste.

        Parâmetros:
        -----------
        n: int
            Quantidade de dados que queremos no conjunto de teste
        """

        K = self._K
        return self._matriz_entrada[(n-K):], self._matriz_saida[(n-K):]

    def criar_vetores(self, indice):
        """
        Descrição:
        ----------
        Função para criar os vetores de entrada e saída para as etapas de treinamento e teste para uma série temporal.
    
        Parâmetros:
        -----------
        indice: int
            A posicao a partir da qual se deseja prever algum valor
        """

        K, L = self._K, self._L
    
        # checa se a partir da posição atual podemos criar um vetor de amostras dado um K
        if (indice < (K-1)):
            raise ValueError("(indice + 1) =", (indice + 1) , "deve ser maior ou igual a (K - 1) =", (K - 1) , "!")
        
        # checa se o valor que queremos prever (que vai ser armazenado na matriz de saida), está dentro da série temporal
        if ((indice+L) > (len(self._dados)-1)):
            raise ValueError("O passo de predição (L =", L,") somado com o índice atual (indice = ", indice,") não deve estourar o número de dados na série temporal!")
    
        vetor_entrada = np.array(self._dados[(indice-(K-1)):(indice+1)])
        vetor_entrada = np.insert(vetor_entrada, 0, 1) # insere o elemento x^0 no vetor
        vetor_saida = np.array(self._dados[indice+L])
    
        return vetor_entrada, vetor_saida

    def criar_matrizes(self):
        """
        Descrição:
        ----------
        Função para criar as matrizes com os vetores de entrada e saída para as etapas de treinamento e teste.
    
        Parâmetros:
        -----------
        Nenhum
        """
    
        K, L = self._K, self._L
        num_dados = len(self._dados)
    
        for indice in range((K-1), (num_dados-L)):
            vetor_entrada, vetor_saida = self.criar_vetores(indice)
        
            if (len(self._matriz_entrada) == 0):
                self._matriz_entrada = vetor_entrada
            else:
                self._matriz_entrada = np.vstack((self._matriz_entrada, vetor_entrada))
                
            if (len(self._matriz_saida) == 0):
                self._matriz_saida = np.array([vetor_saida])
            else:
                self._matriz_saida = np.vstack((self._matriz_saida, vetor_saida))