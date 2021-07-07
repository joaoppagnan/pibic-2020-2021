# power_spectrum.py

# bibliotecas gerais 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

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

# --------- ESPECTRO DE POTÊNCIAS  --------- #
# --------- mapa de henon --------- #
print("Gerando o espectro de potências para o mapa de Hénon...")
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

n_samples = len(x_henon)
ft_freq_range, ft_discrete = signal.freqz(x_henon)
ft_discrete = np.abs(ft_discrete)
power_sp = ft_discrete**2

fig, ax = plt.subplots()
ax.plot(ft_freq_range, power_sp, color='DarkGreen')
ax.plot(-np.flip(ft_freq_range), np.flip(power_sp), color='DarkGreen')
ax.set_ylabel('$|X(e^{j\Omega})|^2$')
ax.set_xlabel('$\Omega$ $[rad]$')
ax.set_xlim([-1*np.pi, 1*np.pi])
ax.grid(True)
ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -np.pi/4, -np.pi/2, -3*np.pi/4, -np.pi])
ax.set_xticklabels([r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$',
                    r'$-\frac{\pi}{4}$', r'$-\frac{\pi}{2}$', r'$-\frac{3\pi}{4}$', r'$-\pi$'])
ax.set_yscale('log')
sns.despine()
fig.savefig('images/espectro-potencias/power-spectrum-henon.pdf')

# --------- mapa logistico --------- #
print("Gerando o espectro de potências para o mapa logístico...")
x_inicial = 0.5
n_inicial = 0
n_iteracoes = 5000
r=3.86
estados = np.array([x_inicial, n_inicial])
mapa = logistic.MapaLogistico(estado_inicial=estados, r=r)
estados = mapa.simular(n_iteracoes)
x_log = estados[:, 0]
n_log = estados[:, 1]

n_samples = len(x_log)
ft_freq_range, ft_discrete = signal.freqz(x_log)
ft_discrete = np.abs(ft_discrete)
power_sp = ft_discrete**2

fig, ax = plt.subplots()
ax.plot(ft_freq_range, power_sp, color='Crimson')
ax.plot(-np.flip(ft_freq_range), np.flip(power_sp), color='Crimson')
ax.set_ylabel('$|X(e^{j\Omega})|^2$')
ax.set_xlabel('$\Omega$ $[rad]$')
ax.set_xlim([-1*np.pi, 1*np.pi])
ax.grid(True)
ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -np.pi/4, -np.pi/2, -3*np.pi/4, -np.pi])
ax.set_xticklabels([r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$',
                    r'$-\frac{\pi}{4}$', r'$-\frac{\pi}{2}$', r'$-\frac{3\pi}{4}$', r'$-\pi$'])
ax.set_yscale('log')
sns.despine()
fig.savefig('images/espectro-potencias/power-spectrum-logistic.pdf')

# --------- sistema de lorenz --------- #
print("Gerando o espectro de potências para o sistema de Lorenz...")
t_inicial = 0
t_final = 50
dt = 0.01
estado_inicial = np.array([0.1, 0, 0])
sis_lorenz = lorenz.SistemaLorenz(estado_inicial, dt=dt)
solucoes_lorenz, instantes_temporais_lorenz = sis_lorenz.calcular(t_inicial = t_inicial, t_final = t_final)
x_lorenz = solucoes_lorenz[:, 0]

n_samples = len(x_lorenz)
ft_freq_range, ft_discrete = signal.freqz(x_lorenz)
ft_discrete = np.abs(ft_discrete)
power_sp = ft_discrete**2

fig, ax = plt.subplots()
ax.plot(ft_freq_range, power_sp, color='DarkBlue')
ax.plot(-np.flip(ft_freq_range), np.flip(power_sp), color='DarkBlue')
ax.set_ylabel('$|X(e^{j\Omega})|^2$')
ax.set_xlabel('$\Omega$ $[rad]$')
ax.set_xlim([-1*np.pi, 1*np.pi])
ax.grid(True)
ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -np.pi/4, -np.pi/2, -3*np.pi/4, -np.pi])
ax.set_xticklabels([r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$',
                    r'$-\frac{\pi}{4}$', r'$-\frac{\pi}{2}$', r'$-\frac{3\pi}{4}$', r'$-\pi$'])
ax.set_yscale('log')
sns.despine()
fig.savefig('images/espectro-potencias/power-spectrum-lorenz.pdf')

# --------- equações de mackey-glass --------- #
print("Gerando o espectro de potências para as equações de Mackey-Glass...")
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

n_samples = len(x_mackeyglass)
ft_freq_range, ft_discrete = signal.freqz(x_mackeyglass)
ft_discrete = np.abs(ft_discrete)
power_sp = ft_discrete**2

fig, ax = plt.subplots()
ax.plot(ft_freq_range, power_sp, color='DarkOrange')
ax.plot(-np.flip(ft_freq_range), np.flip(power_sp), color='DarkOrange')
ax.set_ylabel('$|X(e^{j\Omega})|^2$')
ax.set_xlabel('$\Omega$ $[rad]$')
ax.set_xlim([-1*np.pi, 1*np.pi])
ax.grid(True)
ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -np.pi/4, -np.pi/2, -3*np.pi/4, -np.pi])
ax.set_xticklabels([r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$',
                    r'$-\frac{\pi}{4}$', r'$-\frac{\pi}{2}$', r'$-\frac{3\pi}{4}$', r'$-\pi$'])
ax.set_yscale('log')
sns.despine()
fig.savefig('images/espectro-potencias/power-spectrum-mackeyglass.pdf')

print("Gráficos gerados!")