import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


def Covid_Ode(P, t):
    S, I, Q, R = P
    dS_dt = LAMBDA - mu*S - beta*S*I
    dI_dt = beta*S*I - (delta + mu + alpha_1 + gamma)*I
    dQ_dt = delta*I - epsilon*Q - alpha_2*Q - mu*Q
    dR_dt = gamma*I + epsilon*Q - mu*R
    return [dS_dt, dI_dt, dQ_dt, dR_dt]


def plot_trajetories(P0, t):
    sol = odeint(Covid_Ode, P0, t)
    S_plot = sol[:, 0]
    I_plot = sol[:, 1]
    Q_plot = sol[:, 2]
    R_plot = sol[:, 3]
    return S_plot, I_plot, Q_plot, R_plot

# Parameters


LAMBDA = 1.43
mu = 1.03
beta = 5
delta = 0.4
alpha_1 = 0.42
gamma = 1.05
epsilon = 0.6
alpha_2 = 0.4

# Estudo das Trajetórias Dinâmicas de SIQR

t = np.linspace(0, 100, 1000)

P0 = [1.44, 0.0056, 0.046, 0.05]

S_plot, I_plot, Q_plot, R_plot = plot_trajetories(P0, t)

plt.plot(t, S_plot)
plt.xlabel('t')
plt.ylabel('S(t)')
plt.xlim(0, 70)
plt.ylim(0, 1)
plt.title('Susceptible Population')
plt.show()

plt.plot(t, I_plot)
plt.xlabel('t')
plt.ylabel('I(t)')
plt.xlim(0, 70)
plt.ylim(0, 1)
plt.title('Infected Population')
plt.show()

plt.plot(t, Q_plot)
plt.xlabel('t')
plt.ylabel('Q(t)')
plt.xlim(0, 70)
plt.ylim(0, 0.1)
plt.title('Quarantined Population')
plt.show()

plt.plot(t, R_plot)
plt.xlabel('t')
plt.ylabel('R(t)')
plt.xlim(0, 70)
plt.ylim(0, 0.5)
plt.title('Recovered Population')
plt.show()

# Variações dos Parâmetros

# Delta - Transmission rate from I to Q

delta_vector = np.linspace(0, 10, 100)
print(delta_vector)
