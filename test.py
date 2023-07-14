import math

import matplotlib.pyplot as plt
from scipy.integrate import odeint


def sistema_edo(y, t):
    # Defina aqui as suas equações diferenciais
    dydt = -y
    return dydt


# Condição inicial
y0 = 1

# Pontos de tempo onde você deseja obter as soluções
t = range(0, 10, 1)

# Lista para armazenar as soluções
solucoes = []
f = []
tgraf = []
for i in range(10):
    f.append(math.exp(-t[i]))
# Resolver as EDOs para cada ponto de tempo
for i in range(2):
    sol = odeint(sistema_edo, y0, t)  # Resolver a EDO no intervalo
    # y0 = sol[-1]  # Atualizar a condição inicial para o próximo intervalo
    # print(y0)
    solucoes.extend(sol)  # Adicionar as soluções ao resultado global
    y0 = 2

# As soluções estão armazenadas na lista solucoes

plt.plot(solucoes, 'ro', t, f, 'b')
plt.show()
