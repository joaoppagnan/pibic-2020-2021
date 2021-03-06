# lorenz.py

import numpy as np
from scipy.integrate import odeint

class SistemaLorenz:
    
    def __init__(self, estado_inicial, sigma=10, beta=8/3, rho=28, dt=0.0001):
        """
        Descrição:
        ----------
        Construtor da classe 'SistemaLorenz'

        Parâmetros:
        -----------
        estado_inicial: np.ndarray
            Vetor das posições xyz iniciais do sistema        
        sigma: int ou float
            Parâmetro do Sistema de Lorenz
        beta: int ou float
            Parâmetro do Sistema de Lorenz
        rho: int ou float
            Parâmetro do Sistema de Lorenz
        dt: float
            Tamanho do diferencial de tempo que iremos utilizar nos cálculos, ou seja, a resolução temporal de nossa solução
            
        Retorna:
        --------
        Nada
        """
        
        if not (type(estado_inicial) is np.ndarray):
            raise TypeError("O vetor estado inicial deve ser um array do numpy!") 
        
        if not (((type(sigma) is int) | (type(sigma) is float)) & (sigma > 0)):
            raise TypeError("Sigma deve ser um int ou float positivo!")

        if not (((type(beta) is int) | (type(beta) is float)) & (beta > 0)):
            raise TypeError("Beta deve ser um int ou float positivo!")

        if not (((type(rho) is int) | (type(rho) is float)) & (rho > 0)):
            raise TypeError("Rho deve ser um int ou float positivo!")
            
        if not ((type(dt) is float) & (dt > 0)):
            raise TypeError("dt deve ser um float positivo!")
        
        self._estado_inicial = estado_inicial
        self._sigma = sigma
        self._beta = beta
        self._rho = rho
        self._dt = dt
        pass
    
    def _equacoes(self, estado_atual, t):
        """
        Descrição:
        ----------
        Função interna que retorna as equações do Sistema de Lorenz calculadas no instante t atual
        
        Parâmetros:
        -----------
        estado_atual: np.ndarray
            Vetor das posições xyz atuais do sistema    
        t: float
            Instante t no qual estamos calculando as derivadas
            
        Retorna:
        --------
        Um sistema de EDOs de Lorenz para os parâmetros estimados na forma de um array
        """
        
        if not (type(estado_atual) is np.ndarray):
            raise TypeError("O vetor estado atual deve ser um array do numpy!")        
        
        if not (type(t) is float):
            raise TypeError("t deve ser um float!")
        
        sigma = self._sigma
        beta = self._beta
        rho = self._rho
        x, y, z = estado_atual
        
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        return [dx_dt, dy_dt, dz_dt]
    
    def calcular(self, t_inicial, t_final):
        """
        Descrição:
        ----------
        Evolui o sistema de Lorenz com base nas condições iniciais configuradas, para um t indo de t_inicial até t_final, com n_instantes calculados
        
        Parâmetros:
        -----------
        t_inicial: int
            Instante temporal em que iniciamos os cálculos
        t_final: int
            Instante temporal em que terminamos os cálculos
            
        Retorna:
        --------
        Um vetor com as soluções estimadas e um vetor com os instantes temporais utilizados
        """
        
        if not ((type(t_inicial) is int) & (t_inicial >= 0)):
            raise TypeError("t_inicial deve ser um int não nulo!")
            
        if not ((type(t_final) is int) & (t_final > 0)):
            raise TypeError("t_final deve ser um int positivo!")        
        
        estado_inicial = self._estado_inicial
        dt = self._dt
        
        n_instantes = int((t_final - t_inicial)/dt)
        instantes_temporais = np.linspace(t_inicial, t_final, n_instantes)
        
        solucoes = odeint(self._equacoes, t=instantes_temporais, y0=estado_inicial)
        return solucoes, instantes_temporais