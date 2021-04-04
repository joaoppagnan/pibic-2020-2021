# henonmap.py

import numpy as np

class MapaHenon:

    def __init__(self, a, b, posicao_inicial):
        """
        Descrição:
        ----------
        Construtor da classe 'MapaHenon'

        Parâmetros:
        -----------
        a: float
            Parâmetro do Mapa de Hénon
        b: float
            Parâmetro do Mapa de Hénon
        posicao_inicial: np.ndarray
            Parâmetro das posições xy iniciais do mapa
        """

        if not ((type(a) is float) & (type(b) is float)):
            raise TypeError("Os parâmetros devem ser floats!")
            
        if not (type(posicao_inicial) is np.ndarray):
            raise TypeError("O vetor posição inicial deve ser um array do numpy!")

        self.__a = a
        self.__b = b
        self._x_atual = posicao_inicial[0]
        self._y_atual = posicao_inicial[1]
        pass

    def iterar(self):
        """
        Descrição:
        ----------
        Aplicar as equações de diferenças do Mapa e Hénon para os pontos e parâmetrs atuais

        Parâmetros:
        -----------
        Nenhum
        """

        a = self.__a
        b = self.__b
        x = self._x_atual
        y = self._y_atual

        prox_x = 1 - a*(x**2) + y
        prox_y = b*x
        self._x_atual = prox_x
        self._y_atual = prox_y
        pass

    def posicao(self):
        """
        Descrição:
        ----------
        Retorna a posição x,y atual do mapa

        Parâmetros:
        -----------
        Nenhum
        """

        x = self._x_atual
        y = self._y_atual

        posicao = np.array([x, y])
        return posicao
