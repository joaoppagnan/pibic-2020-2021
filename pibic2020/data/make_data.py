# importando as bibliotecas
import numpy as np
import pandas as pd

# sistemas caoticos
from pibic2020.data import henon
from pibic2020.data import logistic
from pibic2020.data import lorenz
from pibic2020.data import mackeyglass

# gera para o mapa de henon
print("Gerando para o mapa de Hénon...")
mapa_henon = henon.MapaHenon()
n_iteracoes = 5000
dados_henon = mapa_henon.simular(n_iteracoes)
dados_henon = pd.DataFrame(dados_henon, columns=['x', 'y', 'n'])
dados_henon.to_csv('data/raw/henon.csv', index = False, header=True)

# gera para o mapa logistico
print("Gerando para o mapa logístico...")
mapa_logistico = logistic.MapaLogistico()
n_iteracoes = 5000
dados_logistic = mapa_logistico.simular(n_iteracoes)
dados_logistic = pd.DataFrame(dados_logistic, columns=['x', 'n'])
dados_logistic.to_csv('data/raw/logistic.csv', index = False, header=True)

# gera para o sistema de lorenz
print("Gerando para o sistema de Lorenz...")
sistema_lorenz = lorenz.SistemaLorenz(estado_inicial=np.array([0.1, 0, 0]), dt=0.01)
t_inicial = 0
t_final = 50
dados_lorenz, instantes_temporais = sistema_lorenz.calcular(t_inicial=t_inicial, t_final=t_final)
dados_lorenz = np.column_stack((instantes_temporais, dados_lorenz))
dados_lorenz = pd.DataFrame(dados_lorenz, columns=['t', 'x', 'y', 'z'])
dados_lorenz.to_csv('data/raw/lorenz.csv', index = False, header=True)

# gera para as equações de mackey-glass
print("Gerando para as equações de Mackey-Glass")
t_inicial = 0
t_final = 5000
tau = 22
n = 10
gamma = 0.1
beta = 0.2
theta = 1
mackeyglass_eq = mackeyglass.MackeyGlass(tau=tau, gamma=gamma, beta=beta, n=n, theta=theta)
dados_mackeyglass, instantes_temporais = mackeyglass_eq.calcular(t_inicial=t_inicial, t_final=t_final)
dados_mackeyglass = np.column_stack((instantes_temporais, dados_mackeyglass))
dados_mackeyglass = pd.DataFrame(dados_mackeyglass, columns=['t', 'p'])
dados_mackeyglass.to_csv('data/raw/mackeyglass.csv', index = False, header=True)

print("Dados gerados!")