# lorenzsystem.py

import numpy as np

class SistemaLorenz:
    
    def __init__(self, sigma=10, beta=8/3, rho=28):
        """
        Descrição:
        ----------
        Construtor da classe 'SistemaLorenz'

        Parâmetros:
        -----------
        sigma: int ou float
            Parâmetro do Sistema de Lorenz
        beta: int ou float
            Parâmetro do Sistema de Lorenz
        rho: int ou float
            Parâmetro do Sistema de Lorenz
        estado_inicial: np.ndarray
            Vetor das posições xyz iniciais do sistema
        """
        
        if not (((type(sigma) is int) | (type(sigma) is float)) & (sigma > 0)):
            raise TypeError("Sigma deve ser um int ou float positivo!")

        if not (((type(beta) is int) | (type(beta) is float)) & (beta >= 0)):
            raise TypeError("Beta deve ser um int ou float positivo!")

        if not (((type(rho) is int) | (type(rho) is float)) & (rho >= 0)):
            raise TypeError("Rho deve ser um int ou float positivo!")
        
        self._sigma = sigma
        self._beta = beta
        self._rho = rho
        pass
    
    def equacoes(self, estado_atual, t):
        """
        Descrição:
        ----------
        Retorna as equações do Sistema de Lorenz calculadas no instante t atual
        
        Parâmetros:
        -----------
        estado_inicial: np.ndarray
            Vetor das posições xyz atuais do sistema
        """
        
        if not (type(estado_atual) is np.ndarray):
            raise TypeError("O vetor estado atual deve ser um array do numpy!")        
        
        sigma = self._sigma
        beta = self._beta
        rho = self._rho
        x, y, z = estado_atual
        
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        return [dx_dt, dy_dt, dz_dt]