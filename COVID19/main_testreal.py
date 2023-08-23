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


def jacobian_COVID(S, I, mu, beta, delta, alpha_1, gamma, epsilon, alpha_2):
    J = np.zeros([4, 4])
    J[0, 0] = - mu - beta*I
    J[0, 1] = - beta*S
    J[1, 0] = beta*I
    J[1, 1] = beta*S - (delta + mu + alpha_1 + gamma)
    J[2, 1] = delta
    J[2, 2] = - epsilon - alpha_2 - mu
    J[3, 1] = gamma
    J[3, 2] = epsilon
    J[3, 3] = - mu
    return J

# Parameters


LAMBDA = 1.43
mu = -0.2027585585585586
beta = 5
delta = 0.4
alpha_1 = 0.42
gamma = 1.05
epsilon = 0.6
alpha_2 = 0.4

See = LAMBDA/mu
Iee: float = 0
Qee: float = 0
Ree: float = 0

J = jacobian_COVID(See, Iee, mu, beta, delta,
                   alpha_1, gamma, epsilon, alpha_2)

eig_val = np.linalg.eig(J)[0]
print(See, Iee, Qee, Ree)
print(eig_val)

See = (delta + mu + alpha_1 + gamma)/beta
Iee = LAMBDA/(See*beta) - mu/beta
Qee = delta*Iee/(epsilon + alpha_2 + mu)
Ree = (gamma*Iee + epsilon*Qee)/mu

J = jacobian_COVID(See, Iee, mu, beta, delta,
                   alpha_1, gamma, epsilon, alpha_2)

eig_val = np.linalg.eig(J)[0]
print(See, Iee, Qee, Ree)
print(eig_val)
print(f'{mu}\n')
