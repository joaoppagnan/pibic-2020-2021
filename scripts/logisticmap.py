# logisticmap.py

import numpy as np

class MapaLogistico:

    def __init__(self, r, estado_inicial):
        """
        Descrição:
        ----------
        Construtor da classe 'MapaHenon'

        Parâmetros:
        -----------
        r: float
            Parâmetro do Mapa Logístico representando a taxa de crescimento populacional. Deve ser entre 0.0 e 4.0.
        estado_inicial: np.ndarray
            Parâmetro da população x inicial do mapa e do instante n inicial (deve ser 0)
        """

        if not ((type(r) is float) & ((r >= 0) & (r <= 4) )):
            raise TypeError("A taxa de crescimento populacional deve ser um float entre 0.0 e 4.0!")
            
        if not (type(estado_inicial) is np.ndarray):
            raise TypeError("O vetor estado inicial deve ser um array do numpy!")            
            
        if not ((type(estado_inicial[0]) is np.float64) & ((estado_inicial[0] >= 0) & (estado_inicial[0] <= 1))):
            raise TypeError("A população inicial deve ser um float entre 0 e 1!")
            
        if not (estado_inicial[1] == 0):
            raise ValueError("O instante inicial deve ser igual a 0!")    

        self._r = r
        self._x_atual = estado_inicial[0]
        self._n_atual = estado_inicial[1]
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
        
        r = self._r
        x = self._x_atual
        n = self._n_atual

        prox_x = r*x*(1 - x)
        self._x_atual = prox_x
        self._n_atual = n + 1
        pass
    
    def atualizar_r(self, r):
        """
        Descrição:
        ----------
        Atualizar a taxa de crescimento do mapa
        
        Parâmetros:
        -----------
        r: float
            Parâmetro do Mapa Logístico representando a taxa de crescimento populacional. Deve ser entre 0.0 e 4.0.
        """
        
        if not ((type(r) is float) & ((r >= 0) & (r <= 4) )):
            raise TypeError("A taxa de crescimento populacional deve ser um float entre 0.0 e 4.0!")
        
        self._r = r
        pass

    def atualizar_estado(self, estado):
        """
        Descrição:
        ----------
        Atualizar o estado do mapa (sem aplicar a regra)
        
        Parâmetros:
        -----------
        estado_inicial: np.ndarray
            Parâmetro da população x do mapa e do instante n 
        """
        
        if not (type(estado) is np.ndarray):
            raise TypeError("O vetor estado deve ser um array do numpy!")            
            
        if not ((type(estado[0]) is np.float64) & ((estado[0] >= 0) & (estado[0] <= 1))):
            raise TypeError("A população deve ser um float entre 0 e 1!")
        
        self._x_atual = estado[0]
        self._n_atual = estado[1]
        pass    

    def ler_estado(self):
        """
        Descrição:
        ----------
        Retorna o estado atual (x, n) do mapa

        Parâmetros:
        -----------
        Nenhum
        """

        x = self._x_atual
        n = self._n_atual
        
        estado = np.array([x, n])
        return estado