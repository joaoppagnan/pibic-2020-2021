# henonmap.py

import numpy as np

class MapaHenon:

    def __init__(self, estado_inicial=np.array([1, 0, 0]), a=1.4, b=0.3):
        """
        Descrição:
        ----------
        Construtor da classe 'MapaHenon'

        Parâmetros:
        -----------
        estado_inicial: np.ndarray
            Parâmetro das posições xy iniciais do mapa e do instante n inicial (deve ser 0)
        a: float
            Parâmetro do Mapa de Hénon
        b: float
            Parâmetro do Mapa de Hénon
            
        Retorna:
        --------
        Nada
        """

        if not ((type(a) is float) & (type(b) is float)):
            raise TypeError("Os parâmetros devem ser floats!")
            
        if not (type(estado_inicial) is np.ndarray):
            raise TypeError("O vetor estado inicial deve ser um array do numpy!")
            
        if not (estado_inicial[2] == 0):
            raise ValueError("O instante inicial deve ser igual a 0!")

        self.__a = a
        self.__b = b
        self._x_atual = estado_inicial[0]
        self._y_atual = estado_inicial[1]
        self._n_atual = estado_inicial[2]
        self._vetor_estados = estado_inicial
        pass

    def simular(self, n_iteracoes=5000):
        """
        Descrição:
        ----------
        Função para simular o Mapa de Hénon para n_iteracoes

        Parâmetros:
        -----------
        n_iteracoes: int
            Número de iterações da simulação, deve ser maior que 0

        Retorna: 
        --------
        Vetor com os estados para cada n
        """

        if not ((type(n_iteracoes) is int) and (n_iteracoes > 0)):
            raise ValueError("O número de iterações é um inteiro positivo!")

        vetor_estados = self._vetor_estados

        for n in range(0, n_iteracoes):
            self._iterar()
            vetor_estados = np.vstack((vetor_estados, self._ler_estado()))

        return vetor_estados

    def _iterar(self):
        """
        Descrição:
        ----------
        Aplicar as equações de diferenças do Mapa e Hénon para os pontos e parâmetrs atuais

        Parâmetros:
        -----------
        Nenhum
        
        Retorna:
        --------
        Nada
        """

        a = self.__a
        b = self.__b
        x = self._x_atual
        y = self._y_atual
        n = self._n_atual

        prox_x = 1 - a*(x**2) + y
        prox_y = b*x
        self._x_atual = prox_x
        self._y_atual = prox_y
        self._n_atual = n + 1
        pass

    def _ler_estado(self):
        """
        Descrição:
        ----------
        Retorna o estado atual (x, y, n) do mapa

        Parâmetros:
        -----------
        Nenhum
        
        Retorna:
        --------
        Um np.ndarray com as coordenadas (x, y, n) atuais do mapa
        """

        x = self._x_atual
        y = self._y_atual
        n = self._n_atual

        estado = np.array([x, y, n])
        return estado