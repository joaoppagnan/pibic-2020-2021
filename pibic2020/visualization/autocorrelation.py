# autocorrelation.py

# bibliotecas gerais 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# para exibir as autocorrelacoes
from statsmodels.graphics.tsaplots import plot_acf 
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib.collections import PolyCollection, LineCollection

# agora, melhoramos a qualidade de saida e de visualizacao da imagem 
# alem de mudar a fonte padrao para uma do latex
sns.set_style("ticks")
plt.rcParams['savefig.dpi'] = 200
plt.rcParams["figure.dpi"] = 100

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

# sistemas caóticos
from pibic2020.data import logistic
from pibic2020.data import henon
from pibic2020.data import lorenz
from pibic2020.data import mackeyglass

# --------- AUTOCORRELAÇÕES  --------- #
# --------- mapa de henon --------- #
print("Gerando as autocorrelações para o mapa de Hénon...")
a = 1.4
b = 0.3
x_inicial = 1.0
y_inicial = 0.0
n_inicial = 0
estados = np.array([x_inicial, y_inicial, n_inicial])
mapa = henon.MapaHenon(estado_inicial=estados)
n_iteracoes = 5000
estados = mapa.simular(n_iteracoes)
x_henon = estados[:, 0]
y_henon = estados[:, 1]
n_henon = estados[:, 2]

fig, ax = plt.subplots()
cor = '#002d04'
plot_acf(x_henon, ax=ax, color=cor, title=None)
#ax.set_title("Autocorrelação da série temporal em $\hat{x}$ do Mapa de Hénon para\n $a =$ " + str(a) + " e $b =$ " + str(b) + " com $x[0] =$ " + str(x_henon[0]) + " e $y[0] =$ " + str(y_henon[0]))
ax.set_ylabel('Autocorrelação $[K]$')
ax.set_xlabel('$K$')
ax.set_xlim(0,)
for item in ax.collections:
    if type(item)==PolyCollection:
        item.set_facecolor(cor)
    if type(item)==LineCollection:
        item.set_color(cor)    
for item in ax.lines:
    item.set_color(cor)
plt.subplots_adjust(top=0.95)
ax.grid(False)
sns.despine()
fig.savefig("images/autocorrelacao/autocorrelacao-henon.pdf")

fig, ax = plt.subplots()
plot_pacf(x_henon, ax=ax, color=cor, title=None)
#ax.set_title("Autocorrelação parcial da série temporal em $\hat{x}$ do Mapa de Hénon para\n $a =$ " + str(a) + " e $b =$ " + str(b) + " com $x[0] =$ " + str(x_henon[0]) + " e $y[0] =$ " + str(y_henon[0]))
ax.set_ylabel('Autocorrelação parcial $[K]$')
ax.set_xlabel('$K$')
ax.set_xlim(0,)
for item in ax.collections:
    if type(item)==PolyCollection:
        item.set_facecolor(cor)
    if type(item)==LineCollection:
        item.set_color(cor)    
for item in ax.lines:
    item.set_color(cor)
plt.subplots_adjust(top=0.95)
ax.grid(False)
sns.despine()
fig.savefig("images/autocorrelacao/autocorrelacao-parcial-henon.pdf")

# --------- mapa logistico --------- #
print("Gerando as autocorrelações para o mapa logístico...")
x_inicial = 0.5
n_inicial = 0
n_iteracoes = 5000
r=3.86
estados = np.array([x_inicial, n_inicial])
mapa = logistic.MapaLogistico(estado_inicial=estados, r=r)
estados = mapa.simular(n_iteracoes)
x_log = estados[:, 0]
n_log = estados[:, 1]

fig, ax = plt.subplots()
cor = '#4b0101'
plot_acf(x_log, ax=ax, color=cor, title=None)
#ax.set_title("Autocorrelação da série temporal do mapa logístico\n para $r = $ " + str(r) + " e $x[0] =$ " + str(x_log[0]))
ax.set_ylabel('Autocorrelação $[K]$')
ax.set_xlabel('$K$')
ax.set_xlim(0,)
for item in ax.collections:
    if type(item)==PolyCollection:
        item.set_facecolor(cor)
    if type(item)==LineCollection:
        item.set_color(cor)    
for item in ax.lines:
    item.set_color(cor)
plt.subplots_adjust(top=0.95)
ax.grid(False)
sns.despine()
fig.savefig("images/autocorrelacao/autocorrelacao-logistic.pdf")

fig, ax = plt.subplots()
plot_pacf(x_log, ax=ax, color=cor, title=None)
#ax.set_title("Autocorrelação parcial da série temporal do mapa logístico\n para $r = $ " + str(r) + " e $x[0] =$ " + str(x_log[0]))
ax.set_ylabel('Autocorrelação parcial $[K]$')
ax.set_xlabel('$K$')
ax.set_xlim(0,)
for item in ax.collections:
    if type(item)==PolyCollection:
        item.set_facecolor(cor)
    if type(item)==LineCollection:
        item.set_color(cor)    
for item in ax.lines:
    item.set_color(cor)
plt.subplots_adjust(top=0.95)
ax.grid(False)
sns.despine()
fig.savefig("images/autocorrelacao/autocorrelacao-parcial-logistic.pdf")

# --------- sistema de lorenz --------- #
print("Gerando as autocorrelações para o sistema de Lorenz...")
t_inicial = 0
t_final = 50
dt = 0.01
estado_inicial = np.array([0.1, 0, 0])
sis_lorenz = lorenz.SistemaLorenz(estado_inicial, dt=dt)
solucoes_lorenz, instantes_temporais_lorenz = sis_lorenz.calcular(t_inicial = t_inicial, t_final = t_final)
x_lorenz = solucoes_lorenz[:, 0]

fig, ax = plt.subplots()
cor = '#35063e'
plot_acf(x_lorenz, ax=ax, color=cor, title=None)
#ax.set_title("Autocorrelação da série temporal em $\hat{x}$ do Sistema de Lorenz\n utilizando $\sigma = 10$, " + r"$\beta =\frac{8}{3}$, " + r"$\rho=28$, com " + "$x(0) =$ " + str(estado_inicial[0]) + ", $y(0) = $ " + str(estado_inicial[1]) + " e $z(0) =$ " + str(estado_inicial[2]))
ax.set_ylabel('Autocorrelação $[K]$')
ax.set_xlabel('$K$')
ax.set_xlim(0,)
for item in ax.collections:
    if type(item)==PolyCollection:
        item.set_facecolor(cor)
    if type(item)==LineCollection:
        item.set_color(cor)    
for item in ax.lines:
    item.set_color(cor)
plt.subplots_adjust(top=0.95)
ax.grid(False)
sns.despine()
fig.savefig("images/autocorrelacao/autocorrelacao-lorenz.pdf")

fig, ax = plt.subplots()
plot_pacf(x_lorenz, ax=ax, color=cor, method='ywm', title=None)
#ax.set_title("Autocorrelação parcial da série temporal em $\hat{x}$ do Sistema de Lorenz\n utilizando $\sigma = 10$, " + r"$\beta =\frac{8}{3}$, " + r"$\rho=28$, com " + "$x(0) =$ " + str(estado_inicial[0]) + ", $y(0) = $ " + str(estado_inicial[1]) + " e $z(0) =$ " + str(estado_inicial[2]))
ax.set_ylabel('Autocorrelação parcial $[K]$')
ax.set_xlabel('$K$')
ax.set_xlim(0,)
for item in ax.collections:
    if type(item)==PolyCollection:
        item.set_facecolor(cor)
    if type(item)==LineCollection:
        item.set_color(cor)    
for item in ax.lines:
    item.set_color(cor)
plt.subplots_adjust(top=0.95)
ax.grid(False)
sns.despine()
fig.savefig("images/autocorrelacao/autocorrelacao-parcial-lorenz.pdf")

# --------- equações de mackey-glass --------- #
print("Gerando as autocorrelações para as equações de Mackey-Glass...")
t_inicial = 0
t_final = 5000
tau = 22
n = 10
gamma = 0.1
beta = 0.2
theta = 1
macglass = mackeyglass.MackeyGlass(tau=tau, gamma=gamma, beta=beta, n=n, theta=theta)
solucoes, instantes_temporais = macglass.calcular(t_inicial = t_inicial, t_final = t_final)
x_mackeyglass = np.array(solucoes)

fig, ax = plt.subplots()
cor = '#653700'
plot_acf(x_mackeyglass, ax=ax, color=cor, title=None)
#ax.set_title('Autocorrelação da série temporal da equação de Mackey-Glass para\n' + r'$\tau =$ ' + str(tau) + r', $\beta =$ ' + str(beta) + r', $\gamma =$ ' + str(gamma) + r', $n =$ ' + str(n) + r' e $\theta =$ ' + str(theta) + ' utilizando $P(0) =$ ' + str(0.1*theta))
ax.set_ylabel('Autocorrelação $[K]$')
ax.set_xlabel('$K$')
ax.set_xlim(0,)
for item in ax.collections:
    if type(item)==PolyCollection:
        item.set_facecolor(cor)
    if type(item)==LineCollection:
        item.set_color(cor)
for item in ax.lines:
    item.set_color(cor)
plt.subplots_adjust(top=0.95)
ax.grid(False)
sns.despine()
fig.savefig("images/autocorrelacao/autocorrelacao-mackeyglass.pdf")

fig, ax = plt.subplots()
plot_pacf(x_mackeyglass, ax=ax, color=cor, title=None)
#ax.set_title('Autocorrelação parcial da série temporal da equação de Mackey-Glass para\n' + r'$\tau =$ ' + str(tau) + r', $\beta =$ ' + str(beta) + r', $\gamma =$ ' + str(gamma) + r', $n =$ ' + str(n) + r' e $\theta =$ ' + str(theta) + ' utilizando $P(0) =$ ' + str(0.1*theta))
ax.set_ylabel('Autocorrelação parcial $[K]$')
ax.set_xlabel('$K$')
ax.set_xlim(0,)
for item in ax.collections:
    if type(item)==PolyCollection:
        item.set_facecolor(cor)
    if type(item)==LineCollection:
        item.set_color(cor)
for item in ax.lines:
    item.set_color(cor)
plt.subplots_adjust(top=0.95)
ax.grid(False)
sns.despine()
fig.savefig("images/autocorrelacao/autocorrelacao-parcial-mackeyglass.pdf")

print("Gráficos gerados!")