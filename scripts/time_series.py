# io_gen.py

import numpy as np

class SerieTemporal:

    def __init__(self, dados):
        """
        Descrição:
        ----------
        Construtor da classe 'SerieTemporal'

        Parâmetros:
        -----------
        dados: array
                Conjunto de valores da série temporal
        """

        self.__dados = dados
        self.matriz_entrada = np.array([])
        self.matriz_saida = np.array([])


    def criar_vetores(self, K, L, n):
        """
        Descrição:
        ----------
        Função para criar os vetores de entrada e saída para as etapas de treinamento e teste para uma série temporal.
    
        Parâmetros:
        -----------
        dados: array
            Conjunto de dados que você quer utilizar para criar os vetores
        K: int
            O número de entradas utilizado para a predição
        L: int
            O passo de predição 
        n: int
            A posicao a partir da qual se deseja prever algum valor
        """
    
        # checa se a partir da posição atual podemos criar um vetor de amostras dado um K
        if (n < (K-1)):
            raise ValueError("n + 1 deve ser maior ou igual a (K-1)!")
        
        # checa se o valor que queremos prever (que vai ser armazenado na matriz de saida), está dentro da série temporal
        if ((n+L) > (len(self.__dados)-1)):
            raise ValueError("O passo de predição (L =", L,") somado com o índice atual (n = ", n,") não deve estourar o número de dados na série temporal!")
    
        vetor_entrada = np.array(self.__dados[(n-(K-1)):(n+1)])
        vetor_entrada = np.insert(vetor_entrada, 0, 1) # insere o elemento x^0 no vetor
        vetor_saida = np.array(self.__dados[n+L])
    
        return vetor_entrada, vetor_saida

    def criar_matrizes(self, K, L):
        """
        Descrição:
        ----------
        Função para criar as matrizes com os vetores de entrada e saída para as etapas de treinamento e teste.
    
        Parâmetros:
        -----------
        serie_temporal: np.array
            Série temporal que será utilizada para criar os vetores
        K: int
            O número de entradas utilizado para a predição
        L: int
            O passo de predição 
        """
    
        if (K <= 0):
            raise ValueError("O hiperparâmetro K deve ser um inteiro positivo!")
    
        num_dados = len(self.__dados)
    
        for n in range((K-1), (num_dados-L)):
            vetor_entrada, vetor_saida = self.criar_vetores(K, L, n)
        
            if (len(self.matriz_entrada) == 0):
                self.matriz_entrada = vetor_entrada
            else:
                self.matriz_entrada = np.vstack((self.matriz_entrada, vetor_entrada))
                
            if (len(self.matriz_saida) == 0):
                self.matriz_saida = np.array([vetor_saida])
            else:
                self.matriz_saida = np.vstack((self.matriz_saida, vetor_saida))