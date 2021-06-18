# mackeyglassequations.py

import numpy as np
from jitcdde import jitcdde, y, t

class MackeyGlass:
    
    def __init__(self, inicializacao='padrao', tau=6, gamma=0.1, beta=0.2, theta=1, n=10, dt=1.0):
        """
        Descrição:
        ----------
        Construtor da classe 'MackeyGlass'

        Parâmetros:
        -----------
        inicializacao: str
            String referente ao tipo de valores iniciais utilizado para as equações. 
            Se for 'padrao', utiliza a mesma usada no paper de Mackey e Glass, ie, 0.1 para todos os instantes anteriores ao início da medição.
            Caso seja 'aleatoria', utiliza valores aleatorios para esses instantes anteriores.
        tau: int
            Parâmetro das Equações de Mackey-Glass
        gamma: int ou float
            Parâmetro das Equações de Mackey-Glass
        beta: int ou float
            Parâmetro das Equações de Mackey-Glass
        theta: int ou float
            Parâmetro das Equações de Mackey-Glass
        n: int ou float
            Parâmetro das Equações de Mackey-Glass
        dt: float
            Tamanho do diferencial de tempo que iremos utilizar nos cálculos, ou seja, a resolução temporal de nossa solução
            
        Retorna:
        --------
        Nada
        """
        
        if not (type(inicializacao) is str):
            raise TypeError("O tipo da inicialização deve ser uma string!")
        
        if not (((type(gamma) is int) | (type(gamma) is float)) & (gamma > 0)):
            raise TypeError("Gamma deve ser um int ou float positivo!")

        if not (((type(beta) is int) | (type(beta) is float)) & (beta > 0)):
            raise TypeError("Beta deve ser um int ou float positivo!")

        if not (((type(theta) is int) | (type(theta) is float)) & (theta > 0)):
            raise TypeError("Theta deve ser um int ou float positivo!")
                
        if not (((type(tau) is int) | (type(tau) is float)) & (tau > 0)):
            raise TypeError("Tau deve ser um int ou float positivo!")
                
        if not (((type(n) is int) | (type(n) is float)) & (n > 0)):
            raise TypeError("n deve ser um int ou float positivo!")
            
        if not ((type(dt) is float) & (dt > 0)):
            raise TypeError("dt deve ser um float positivo!")
        
        if (inicializacao == 'padrao'):
            p_iniciais = (np.ones((tau, 1))*0.1)*theta
            
        elif (inicializacao == 'aleatoria'):
            p_iniciais = np.random.rand(tau, 1)*theta
            
        else:
            raise ValueError("O tipo da inicialização deve ser um dos dois apresentados(aleatoria ou padrao)!")
        
        self._p_iniciais = p_iniciais
        self._gamma = gamma
        self._beta = beta
        self._theta = theta
        self._tau = tau
        self._n = n
        self._dt = dt
        pass
    
    def _equacao(self):
        """
        Descrição:
        ----------
        Função interna que retorna as equações de Mackey-Glass calculadas no instante t atual
        
        Parâmetros:
        -----------
        Nenhum
        
        Retorna:
        --------
        A equação de Mackey-Glass para os parâmetros estimados na forma de um array
        """     
        
        gamma = self._gamma
        beta = self._beta
        tau = self._tau
        theta = self._theta
        n = self._n
        
        dp_dt = ((beta)*y(0,t-tau)*(theta**n))/((theta**n) + (y(0,t-tau))**n) - gamma*y(0)
        return [dp_dt]
    
    def calcular(self, t_inicial, t_final):
        """
        Descrição:
        ----------
        Evolui as equações de Mackey-Glass com base nas condições iniciais configuradas, para um t indo de t_inicial até t_final,
        
        Parâmetros:
        -----------
        t_inicial: int
            Dia em que iniciamos os cálculos
        t_final: int
            Dia em que terminamos os cálculos
            
        Retorna:
        --------
        Um vetor com as soluções estimadas e um vetor com os instantes temporais utilizados
        """
        
        if not ((type(t_inicial) is int) & (t_inicial >= 0)):
            raise TypeError("t_inicial deve ser um int não nulo!")
            
        if not ((type(t_final) is int) & (t_final > 0)):
            raise TypeError("t_final deve ser um int positivo!")
        
        p_iniciais = self._p_iniciais
        dt = self._dt
        
        p_derivadas_iniciais = np.zeros(len(p_iniciais))
        t_anteriores = np.arange(-len(p_iniciais), 0, 1)
        condicoes_iniciais = []
        
        for i in range(0, len(p_iniciais), 1):
            condicao = (t_anteriores[i], p_iniciais[i], p_derivadas_iniciais[i])
            condicoes_iniciais.append(condicao)
        
        equacao = self._equacao()
        DDE = jitcdde(equacao)
        
        DDE.add_past_points(condicoes_iniciais)
        DDE.step_on_discontinuities()
        DDE.set_integration_parameters(first_step = dt, max_step = dt)
        
        instantes_temporais = np.arange(t_inicial, t_final, dt)
        t_integracao = np.arange(DDE.t + t_inicial, DDE.t + t_final, dt)
        
        solucoes = []
        for t in t_integracao:
            solucoes.append(DDE.integrate(t))
        
        return solucoes, instantes_temporais