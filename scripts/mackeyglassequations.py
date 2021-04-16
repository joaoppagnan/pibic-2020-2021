# mackeyglassequations.py

class MackeyGlass:
    
    def __init__(self, p_inicial, gamma=0.1, beta=0.2, theta=1, tau=30, n=30, dt=0.0001):
        """
        Descrição:
        ----------
        Construtor da classe 'SistemaLorenz'

        Parâmetros:
        -----------
        p_inicial: int ou float
            Valor inicial da concentração de glóbulos vermelhos no sangue
        gamma: int ou float
            Parâmetro das Equações de Mackey-Glass
        beta: int ou float
            Parâmetro das Equações de Mackey-Glass
        theta: int ou float
            Parâmetro das Equações de Mackey-Glass
        tau: int
            Parâmetro das Equações de Mackey-Glass
        n: int
            Parâmetro das Equações de Mackey-Glass
        dt: float
            Tamanho do diferencial de tempo que iremos utilizar nos cálculos, ou seja, a resolução temporal de nossa solução
            
        Retorna:
        --------
        Nada
        """
        
        if not ((type(p_inicial) is int) | (type(p_inicial) is float) & (p_inicial > 0)):
            raise TypeError("0 valor da concentração de glóbulos vermelhos no sangue deve ser um int ou float positivo!")
        
        if not (((type(gamma) is int) | (type(gamma) is float)) & (gamma > 0)):
            raise TypeError("Gamma deve ser um int ou float positivo!")

        if not (((type(beta) is int) | (type(beta) is float)) & (beta > 0)):
            raise TypeError("Beta deve ser um int ou float positivo!")

        if not (((type(theta) is int) | (type(theta) is float)) & (theta > 0)):
            raise TypeError("Theta deve ser um int ou float positivo!")
                
        if not (((type(tau) is int) | (type(tau) is float)) & (tau > 0)):
            raise TypeError("Tau deve ser um int ou float positivo!")
                
        if not ((type(n) is int) & (n > 0)):
            raise TypeError("n deve ser um int positivo!")
            
        if not ((type(dt) is float) & (dt > 0)):
            raise TypeError("dt deve ser um float positivo!")
        
        self._p_inicial = p_inicial
        self._gamma = gamma
        self._beta = beta
        self._theta = theta
        self._tau = tau
        self._n = n
        self._dt = dt
        
        p_delays = np.random.rand(tau, 1)
        self._p_delays = p_delays
        pass
    
    def _equacoes(self, estado_atual, t):
        """
        Descrição:
        ----------
        Função interna que retorna as equações de Mackey-Glass calculadas no instante t atual
        
        Parâmetros:
        -----------
        estado_atual: np.ndarray
            Vetor das posições (p_atual, p_delay) atuais do sistema    
        t: float
            Instante t no qual estamos calculando as derivadas
            
        Retorna:
        --------
        As equações de Mackey-Glass para os parâmetros estimados na forma de um array
        """
        
        if not (type(estado_atual) is np.ndarray):
            raise TypeError("O vetor estado atual deve ser um array do numpy!")        
        
        if not (type(t) is float):
            raise TypeError("t deve ser um float!")
        
        gamma = self._gamma
        beta = self._beta
        theta = self._theta
        tau = self._tau
        n = self._n
        p_atual, p_delay = estado_atual
        
        dp_dt = (beta*((theta)**n))/(((theta)**n) + (p_delay**n)) - gamma*p_atual
        dq_dt = (beta*((theta)**n)*p_delay)/(((theta)**n) + (p_delay**n)) - gamma*p_atual
        return [dp_dt, dq_dt]
    
    def calcular(self, t_inicial, t_final):
        """
        Descrição:
        ----------
        Evolui as equações de Mackey-Glass com base nas condições iniciais configuradas, para um t indo de t_inicial até t_final, com n_instantes calculados
        
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
        
        p_delays = self._p_delays
        dt = self._dt
        
        n_instantes = int((t_final - t_inicial)/dt)        
        instantes_temporais = np.linspace(t_inicial, t_final, n_instantes)
        
        solucoes = odeint(self._equacoes, t=instantes_temporais, y0=estado_inicial)
        return solucoes, instantes_temporais