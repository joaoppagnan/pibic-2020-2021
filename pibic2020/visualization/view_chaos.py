# bibliotecas gerais
import numpy as np

# bibliotecas dos graficos
import matplotlib.pyplot as plt
import seaborn as sns

# agora, melhoramos a qualidade de saida e de visualizacao da imagem 
# alem de mudar a fonte padrao para uma do latex
sns.set_style("ticks")
plt.rcParams['savefig.dpi'] = 200
plt.rcParams["figure.dpi"] = 150
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
#plt.style.use("dark_background")


# bibliotecas dos sistemas caoticos
from pibic2020.data import henon
from pibic2020.data import logistic
from pibic2020.data import lorenz
from pibic2020.data import mackeyglass

print("Escolha o sistema para gerar os gráficos")
print("1: Mapa de Hénon, 2: Mapa Logístico, 3: Sistema de Lorenz, 4: Equações de Mackey-Glass")
sis = input()

# --------------- HENON --------------- #
if (int(sis) == 1):
    a = 1.4
    b = 0.3
    x_inicial = 1.0
    y_inicial = 0.0
    n_inicial = 0
    estado_inicial = np.array([x_inicial, y_inicial, n_inicial])

    mapa_henon = henon.MapaHenon(estado_inicial=estado_inicial, a=a, b=b)
    n_iteracoes = 100000
    vetor_estados = mapa_henon.simular(n_iteracoes)
    x = vetor_estados[:, 0]
    y = vetor_estados[:, 1]
    n = vetor_estados[:, 2]

    fig, ax = plt.subplots()
    ax.scatter(x, y, s=0.01, marker=".", color='Black')
    #ax.set_title(str(n_iteracoes) + " iterações do Mapa de Hénon para $a =$ " + str(a) + " e $b =$ " + str(b) + "\n com $x[0] =$ " + str(x[0]) + " e $y[0] =$ " + str(y[0]))
    ax.set_ylabel('$y$')
    ax.set_xlabel('$x$')
    ax.set_ylim([-0.4, 0.4])
    ax.set_xlim([-1.4, 1.4])
    ax.grid(False)
    sns.despine()
    fig.tight_layout()
    fig.savefig("images/caos/henon-map/mapa-de-henon.pdf")

    fig, ax = plt.subplots(2)
    #fig.suptitle("100 primeiras iterações das séries temporais do Mapa de Hénon para\n $a =$ " + str(a) + " e $b =$ " + str(b) + " com $x[0] =$ " + str(x[0]) + " e $y[0] =$ " + str(y[0]))
    ax[0].plot(n, x, color='Crimson', linewidth=0.9)
    ax[0].set_ylabel('$x[n]$')
    ax[0].set_xlabel('$n$')
    ax[0].set_xlim(0, 100)
    ax[0].grid(True)
    ax[1].plot(n, y, color='DarkGreen', linewidth=0.9)
    ax[1].set_ylabel('$y[n]$')
    ax[1].set_xlabel('$n$')
    ax[1].set_xlim(0, 100)
    ax[1].grid(True)
    sns.despine()
    fig.tight_layout()
    fig.savefig("images/caos/henon-map/series-temporais.pdf")

    fig, ax = plt.subplots()
    ax.plot(n, x, color='DarkGreen', linewidth=0.9)
    #ax.set_title("100 primeiras iterações das séries temporais do Mapa de Hénon em $\hat{x}$ para\n $a =$ " + str(a) + " e $b =$ " + str(b) + " com $x[0] =$ " + str(x[0]) + " e $y[0] =$ " + str(y[0]))
    ax.set_ylabel('$x[n]$')
    ax.set_xlabel('$n$')
    ax.set_xlim(0, 100)
    ax.grid(True)
    sns.despine()
    fig.tight_layout()
    fig.savefig("images/caos/henon-map/series-temporal-x.pdf")
    print("Gráficos gerados!")

# --------------- LOGISTICO --------------- #
elif (int(sis) == 2):
    estado_inicial = np.array([0.5, 0])
    r = 3.86

    mapa_logistico = logistic.MapaLogistico(estado_inicial=estado_inicial, r=r)
    n_iteracoes = 5000
    vetor_estados = mapa_logistico.simular(n_iteracoes)
    x = vetor_estados[:, 0]
    n = vetor_estados[:, 1]

    fig, ax = plt.subplots()
    ax.plot(n, x, color='Crimson', linewidth=0.9)
    #ax.set_title("100 iterações iniciais da série temporal do Mapa Logístico\n para $r =$ " + str(r) + " com $x[0] =$ " + str(x[0]))
    ax.set_ylabel('$x[n]$')
    ax.set_xlabel('$n$')
    ax.set_xlim(0, 100)
    ax.grid(True)
    sns.despine()
    fig.tight_layout()
    fig.savefig("images/caos/logistic-map/serie-temporal.pdf")

    conjunto_r = np.linspace(0.0, 4.0, 1000)
    n_iteracoes = 1000
    n_valores_finais = int(0.1*n_iteracoes)
    fig, ax = plt.subplots()
    #ax.set_title("Diagrama de bifurcação para o mapa logístico")
    ax.set_ylabel('$x$')
    ax.set_xlabel('$r$')
    ax.set_xlim(0, 4)
    ax.set_ylim(0, )
    ax.grid(False)
    sns.despine()
    x_plot = []
    r_plot = []
    mapa_log = logistic.MapaLogistico(r=0.0, estado_inicial=estado_inicial)
    for ri in range(0, len(conjunto_r)):
        estados = estado_inicial
        mapa_log.atualizar_estado(estados)
        mapa_log.atualizar_r(float(conjunto_r[ri]))
    
        vetor_estados = mapa_log.simular(n_iteracoes)
        x = vetor_estados[:, 0]

        x_unicos = np.unique(x[-n_valores_finais:])
        r_unicos = conjunto_r[ri]*np.ones(len(x_unicos))

        ax.scatter(r_unicos, x_unicos, s=0.5, marker="o", facecolors='Black', edgecolors='Black')
    fig.tight_layout()
    fig.savefig("images/caos/logistic-map/mapa-logistico.pdf")
    print("Gráficos gerados!")

# --------------- LORENZ --------------- #
elif (int(sis) == 3):
    t_inicial = 0
    t_final = 50
    dt = 0.001
    estado_inicial = np.array([0.1, 0, 0])

    sis_lorenz = lorenz.SistemaLorenz(estado_inicial=estado_inicial, dt=dt)
    vetor_posicao, instantes_temporais = sis_lorenz.calcular(t_inicial=t_inicial, t_final=t_final)
    x = vetor_posicao[:, 0]
    y = vetor_posicao[:, 1]
    z = vetor_posicao[:, 2]

    fig, ax = plt.subplots()
    ax = plt.axes(projection='3d')
    ax.plot(x, y, z, alpha=1, linewidth=0.6, color='Black')
    #ax.set_title("Diagrama de fases do Atrator de Lorenz\n utilizando $\sigma = 10$, " + r"$\beta =\frac{8}{3}$, " + r"$\rho=28$, com " + "$x(0) =$ " + str(estado_inicial[0]) + ", $y(0) = $ " + str(estado_inicial[1]) + " e $z(0) =$ " + str(estado_inicial[2]))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.w_xaxis.set_pane_color((0, 0, 0, 0))
    ax.w_yaxis.set_pane_color((0, 0, 0, 0))
    ax.w_zaxis.set_pane_color((0, 0, 0, 0))
    ax.grid(True)
    sns.despine()
    fig.tight_layout()
    plt.subplots_adjust(top=1.05)
    fig.savefig("images/caos/lorenz/diagrama-de-fases.pdf")

    fig, ax = plt.subplots(3)
    #fig.suptitle("Séries temporais de 0 a 100 segundos das coordenadas $xyz$ do Sistema de Lorenz\n utilizando $\sigma = 10$, " + r"$\beta =\frac{8}{3}$, " + r"$\rho=28$, com " + "$x(0) =$ " + str(estado_inicial[0]) + ", $y(0) = $ " + str(estado_inicial[1]) + " e $z(0) =$ " + str(estado_inicial[2]))
    ax[0].plot(instantes_temporais, x, color='Crimson', linewidth=0.9)
    ax[0].set_ylabel('$x(t)$')
    ax[0].set_xlabel('$t$')
    ax[0].set_xlim(0,50)  
    ax[0].grid(True)
    ax[1].plot(instantes_temporais, y, color='DarkGreen', linewidth=0.9)
    ax[1].set_ylabel('$y(t)$')
    ax[1].set_xlabel('$t$')
    ax[1].set_xlim(0,50)
    ax[1].grid(True)
    ax[2].plot(instantes_temporais, z, color='DarkBlue', linewidth=0.9)
    ax[2].set_ylabel('$z(t)$')
    ax[2].set_xlabel('$t$')
    ax[2].set_xlim(0,50)
    ax[2].grid(True)
    sns.despine()
    plt.tight_layout()
    fig.savefig("images/caos/lorenz/series-temporais.pdf")

    fig, ax = plt.subplots()
    #fig.suptitle("Série temporal em $\hat{x}$ de 0 a 100 segundos do Sistema de Lorenz\n utilizando $\sigma = 10$, " + r"$\beta =\frac{8}{3}$, " + r"$\rho=28$, com " + "$x(0) =$ " + str(estado_inicial[0]) + ", $y(0) = $ " + str(estado_inicial[1]) + " e $z(0) =$ " + str(estado_inicial[2]))
    ax.plot(instantes_temporais, x, color='DarkBlue', linewidth=0.9)
    ax.set_ylabel('$x(t)$')
    ax.set_xlabel('$t$')
    ax.set_xlim(0,50)  
    ax.grid(True)
    sns.despine()
    fig.tight_layout()
    fig.savefig("images/caos/lorenz/serie-temporal-x.pdf")    
    print("Gráficos gerados!")

# --------------- MACKEYGLASS --------------- #
elif (int(sis) == 4):
    t_inicial = 0
    t_final = 800
    tau = 6
    n = 10
    gamma = 0.1
    beta = 0.2
    theta = 1
    dt = 1.0
    mackeyglass_eq = mackeyglass.MackeyGlass(tau=tau, gamma=gamma, beta=beta, n=n, theta=theta, dt=dt)
    dados, instantes_temporais = mackeyglass_eq.calcular(t_inicial=t_inicial, t_final=t_final)
    fig, ax = plt.subplots()
    #ax.set_title('Série temporal de 0 a 800 dias da equação de Mackey-Glass para\n' + r'$\tau =$ ' + str(tau) + r', $\beta =$ ' + str(beta) + r', $\gamma =$ ' + str(gamma) + r', $n =$ ' + str(n) + r' e $\theta =$ ' + str(theta) + ' utilizando P(0) = ' + str(0.1*theta))
    ax.plot(instantes_temporais, dados, color='DarkOrange', linewidth=0.9)
    ax.set_ylabel('$P(t)$')
    ax.set_xlabel('$t$')
    ax.set_xlim(0,800)
    ax.grid(True)
    sns.despine()
    fig.tight_layout()
    fig.savefig("images/caos/mackeyglass/serie-temporal-1.pdf")
    p_atual_2D = np.array([])
    p_tau_2D = np.array([])   
    for i in range(tau, len(dados)):
        p_atual_2D = np.append(p_atual_2D, dados[i])
        p_tau_2D = np.append(p_tau_2D, dados[i - tau])
    fig, ax = plt.subplots()
    #ax.set_title('Atrator de Mackey-Glass em 2D para ' + r'$\tau =$ ' + str(tau) + r', $\beta =$ ' + str(beta) + r', $\gamma =$ ' + str(gamma) + r', $n =$ ' + str(n) + r' e $\theta =$ ' + str(theta) + ',\n utilizando $P(0) =$ ' + str(0.1*theta))
    ax.plot(p_atual_2D, p_tau_2D, color='Black')
    ax.set_ylabel(r'$P(t - \tau)$')
    ax.set_xlabel('$P(t)$')
    ax.grid(False)
    sns.despine()
    fig.tight_layout()
    fig.savefig("images/caos/mackeyglass/atrator-1-2d.pdf")
    p_atual_3D = np.array([])
    p_tau_3D = np.array([])   
    p_2tau_3D = np.array([])
    for i in range(2*tau, len(dados)):
        p_atual_3D = np.append(p_atual_3D, dados[i])
        p_tau_3D = np.append(p_tau_3D, dados[i - tau])
        p_2tau_3D = np.append(p_2tau_3D, dados[i - 2*tau])
    fig, ax = plt.subplots()
    ax = plt.axes(projection='3d')
    #ax.set_title('Atrator de Mackey-Glass em 3D para ' + r'$\tau =$ ' + str(tau) + r', $\beta =$ ' + str(beta) + r', $\gamma =$ ' + str(gamma) + r', $n =$ ' + str(n) + r' e $\theta =$ ' + str(theta) + ',\n utilizando $P(0) =$ ' + str(0.1*theta))
    ax.plot(p_atual_3D, p_tau_3D, p_2tau_3D, color='Black')
    ax.set_xlabel('$P(t)$')
    ax.set_ylabel(r'$P(t - \tau)$')
    ax.set_zlabel(r'$P(t - 2\tau)$')
    ax.w_xaxis.set_pane_color((0, 0, 0, 0))
    ax.w_yaxis.set_pane_color((0, 0, 0, 0))
    ax.w_zaxis.set_pane_color((0, 0, 0, 0))
    ax.grid(True)
    sns.despine()
    fig.tight_layout()
    plt.subplots_adjust(top=1.05)
    fig.savefig("images/caos/mackeyglass/atrator-1-3d.pdf")

    t_inicial = 0
    t_final = 800
    tau = 22
    n = 10
    gamma = 0.1
    beta = 0.2
    theta = 1
    dt = 1.0
    mackeyglass_eq = mackeyglass.MackeyGlass(tau=tau, gamma=gamma, beta=beta, n=n, theta=theta, dt=dt)
    dados, instantes_temporais = mackeyglass_eq.calcular(t_inicial=t_inicial, t_final=t_final)
    fig, ax = plt.subplots()
    #ax.set_title('Série temporal de 0 a 800 dias da equação de Mackey-Glass em caos para\n' + r'$\tau =$ ' + str(tau) + r', $\beta =$ ' + str(beta) + r', $\gamma =$ ' + str(gamma) + r', $n =$ ' + str(n) + r' e $\theta =$ ' + str(theta) + ' utilizando P(0) = ' + str(0.1*theta))
    ax.plot(instantes_temporais, dados, color='DarkOrange', linewidth=0.9)
    ax.set_ylabel('$P(t)$')
    ax.set_xlabel('$t$')
    ax.set_xlim(0,800)
    ax.grid(True)
    sns.despine()
    fig.tight_layout()
    fig.savefig("images/caos/mackeyglass/serie-temporal-2.pdf")
    p_atual_2D = np.array([])
    p_tau_2D = np.array([])   
    for i in range(tau, len(dados)):
        p_atual_2D = np.append(p_atual_2D, dados[i])
        p_tau_2D = np.append(p_tau_2D, dados[i - tau])
    fig, ax = plt.subplots()
    #ax.set_title('Atrator de Mackey-Glass em caos em 2D para ' + r'$\tau =$ ' + str(tau) + r', $\beta =$ ' + str(beta) + r', $\gamma =$ ' + str(gamma) + r', $n =$ ' + str(n) + r' e $\theta =$ ' + str(theta) + ',\n utilizando $P(0) =$ ' + str(0.1*theta))
    ax.plot(p_atual_2D, p_tau_2D, color='Black')
    ax.set_ylabel(r'$P(t - \tau)$')
    ax.set_xlabel('$P(t)$')
    ax.grid(False)
    sns.despine()
    fig.tight_layout()
    fig.savefig("images/caos/mackeyglass/atrator-2-2d.pdf")
    p_atual_3D = np.array([])
    p_tau_3D = np.array([])   
    p_2tau_3D = np.array([])
    for i in range(2*tau, len(dados)):
        p_atual_3D = np.append(p_atual_3D, dados[i])
        p_tau_3D = np.append(p_tau_3D, dados[i - tau])
        p_2tau_3D = np.append(p_2tau_3D, dados[i - 2*tau])
    fig, ax = plt.subplots()
    ax = plt.axes(projection='3d')
    #ax.set_title('Atrator de Mackey-Glass em caos 3D para ' + r'$\tau =$ ' + str(tau) + r', $\beta =$ ' + str(beta) + r', $\gamma =$ ' + str(gamma) + r', $n =$ ' + str(n) + r' e $\theta =$ ' + str(theta) + ',\n utilizando $P(0) =$ ' + str(0.1*theta))
    ax.plot(p_atual_3D, p_tau_3D, p_2tau_3D, color='Black')
    ax.set_xlabel('$P(t)$')
    ax.set_ylabel(r'$P(t - \tau)$')
    ax.set_zlabel(r'$P(t - 2\tau)$')
    ax.w_xaxis.set_pane_color((0, 0, 0, 0))
    ax.w_yaxis.set_pane_color((0, 0, 0, 0))
    ax.w_zaxis.set_pane_color((0, 0, 0, 0))
    ax.grid(True)
    sns.despine()
    fig.tight_layout()
    plt.subplots_adjust(top=1.05)
    fig.savefig("images/caos/mackeyglass/atrator-2-3d.pdf")
    print("Gráficos gerados!")

else:
    print("Comando inválido!")    