# henonmap.py

class MapaHenon:

    def __init__(self, a, b, x_inicial, y_inicial):
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
        """

        if not ((type(a) is float) & (type(b) is float)):
            raise TypeError("Os parâmetros devem ser floats!")        

        self._a = a
        self._b = b
        self._x_atual = x_inicial
        self._y_atual = y_inicial

    def iterar(self):
        """
        Descrição:
        ----------
        Aplicar as equações de diferenças do Mapa e Hénon para os pontos e parâmetrs atuais

        Parâmetros:
        -----------
        Nenhum
        """

        x_atual = self._x_atual
        y_atual = self._x_atual
        a = self._a
        b = self._b

        prox_x = 1 - a*(x_atual**2) + y_atual
        prox_y = b*x_atual

        self._x_atual = prox_x
        self._y_atual = prox_y

    def posicao(self):
        """
        Descrição:
        ----------
        Retorna a posição x,y atual do mapa

        Parâmetros:
        -----------
        Nenhum
        """

        posicao = np.array([self._x_atual, self._y_atual])
        return posicao
