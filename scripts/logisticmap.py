# logisticmap.py

class MapaLogistico:

    def __init__(self, r, x_inicial):
        """
        Descrição:
        ----------
        Construtor da classe 'MapaHenon'

        Parâmetros:
        -----------
        r: float
            Parâmetro do Mapa Logístico representando a taxa de crescimento populacional. Deve ser entre 0.0 e 4.0.
        x_inicial: float
            População inicial, deve variar entre 0 e 1.
        """

        if not ((type(r) is float) & ((r >= 0) & (r <= 4) )):
            raise TypeError("A taxa de crescimento populacional deve ser um float entre 0.0 e 4.0!")
            
        if not ((type(x_inicial) is float) & ((x_inicial >= 0) & (x_inicial <= 1))):
            raise TypeError("A população inicial deve ser um float!")

        self.__r = r
        self._x_atual = x_inicial
        
        pass

    def iterar(self):
        """
        Descrição:
        ----------
        Aplicar as equações de diferenças do Mapa Logístico para os pontos e parâmetrs atuais

        Parâmetros:
        -----------
        Nenhum
        """
        
        r = self.__r
        x = self._x_atual

        prox_x = r*x*(1 - x)
        self._x_atual = prox_x
        
        pass

    def posicao(self):
        """
        Descrição:
        ----------
        Retorna a posição x do mapa

        Parâmetros:
        -----------
        Nenhum
        """

        x = self._x_atual
        
        posicao = np.array([x])
        return posicao