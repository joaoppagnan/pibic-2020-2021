# input_output_gen.py

import numpy as np

def criar_vetores(dados, K, L, n):
    """
    Descrição:
    ----------
        Função para criar os vetores de entrada e saída para as etapas de treinamento e teste para uma série temporal.
    
    Parâmetros:
    -----------
    dados: np.array
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
        
    if ((n+L) > (len(dados)-1)):
        raise ValueError("O passo de predição (L =", L,") somado com o índice atual (n = ", n,") não deve estourar o número de dados na série temporal!")
    
    vetor_entrada = np.array(dados[(n-(K-1)):(n+1)])
    vetor_entrada = np.insert(vetor_entrada, 0, 1) # insere o elemento x^0 no vetor
    vetor_saida = np.array(dados[n+L])
    
    return vetor_entrada, vetor_saida

def criar_matrizes(serie_temporal, K, L):
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
    
    num_dados = len(serie_temporal)
    
    matriz_entrada = np.array([])
    matriz_saida = np.array([])
    
    for n in range((K-1), (num_dados-L)):
        vetor_entrada, vetor_saida = criar_vetores(serie_temporal, K, L, n)
        
        if (len(matriz_entrada) == 0):
            matriz_entrada = vetor_entrada
        else:
            matriz_entrada = np.vstack((matriz_entrada, vetor_entrada))
                
        if (len(matriz_saida) == 0):
            matriz_saida = np.array([vetor_saida])
        else:
            matriz_saida = np.vstack((matriz_saida, vetor_saida))
        
    return matriz_entrada, matriz_saida