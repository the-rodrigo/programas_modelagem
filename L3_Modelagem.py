import math

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint


def get_Qr(T):
    return (q*Cp + hA)*T - q*Cp*Tf - hA*Tcf


def get_Qg(T):
    return (deltaH_neg*v*q*Cf*k0*math.exp(-E/(R*T)))/(q + v*k0*math.exp(-E/(R*T)))


def get_f(T):
    return get_Qg(T) - get_Qr(T)


def get_C(T):
    return q*Cf/(q + v*k0*math.exp(-E/(R*T)))


def get_df(T):
    return deltaH_neg*v*(q**2)*Cf*k0*E*math.exp(E/(R*T))/((T**2)*((q*math.exp(E/(R*T)) + v*k0)**2)) - q*Cp - hA


def get_newT(T, C):
    return (deltaH_neg*v*k0*C*math.exp(-E/(R*T)) + hA*Tcf + q*Cp*Tf)/(q*Cp + hA)


def get_Tcf(q, T):
    return T + v*(q*Cp*(T - Tf)/hA - deltaH_neg*k0*q*Cf/(q + k0*math.exp(-E/(R*T)))*math.exp(-E/(R*T))/hA)


def set_SR(tolerance, T0):
    error = 1
    C = []
    T = [T0]
    i = 0
    while error > tolerance and i <= 10000:
        C.append(get_C(T[i]))
        T.append(get_newT(T[i], C[i]))
        error = abs((T[i+1]-T[i])/T[i])
        i += 1
    return T[-1], C[-1]


def set_NR(tolerance, T0):
    error = 1
    C = []
    T = [T0]
    i = 0
    while error > tolerance and i <= 10000:
        f = get_f(T[i])
        df = get_df(T[i])
        C.append(get_C(T[i]))
        T.append(T[i] - f/df)
        error = abs((T[i+1]-T[i])/T[i])
        i += 1
    return T[-1], C[-1]

# Parâmetros


q: float = 0.1
v: float = 0.1
k0: float = 9703*3600
deltaH_neg: float = 5960
E: float = 11843
Cp: float = 500
hA: float = 15
R: float = 1.987
Tcf: float = 298.5
Tf: float = 298.15
Cf: float = 10

# Número 1

# Letra A

T_estacionario = []
C_estacionario = []

tolerance = 10**-25

TEE, CEE = set_NR(tolerance, 312)
T_estacionario.append(TEE)
C_estacionario.append(CEE)

TEE, CEE = set_SR(tolerance, 337)
T_estacionario.append(TEE)
C_estacionario.append(CEE)

TEE, CEE = set_SR(tolerance, 367)
T_estacionario.append(TEE)
C_estacionario.append(CEE)

print(T_estacionario, C_estacionario)


def dSdt(S, t):
    C, T = S
    dCdt = q * Cf / v - C * (k0 * math.exp(-E / (R * T)) + q / v)
    dTdt = q * Tf / v + deltaH_neg * k0 * C * \
        math.exp(-E / (R * T)) / Cp + hA * Tcf / \
        (v * Cp) - T * (q / v + hA / (v * Cp))
    return [dCdt, dTdt]


def plot_trajetories(t, T0, C0, line_type='k-', label=None):
    S0 = [C0, T0]
    sol = odeint(dSdt, S0, t)

    C_plot = sol[:, 0]
    T_plot = sol[:, 1]

    plt.plot(C_plot, T_plot, line_type, lw=1, label=label)


def plot_separatrix(t, inf, Tee0, Cee0):
    T0 = Tee0 + inf
    C0 = Cee0
    plot_trajetories(t, T0, C0, 'k--', 'Separatriz')

    T0 = Tee0 - inf
    C0 = Cee0
    plot_trajetories(t, T0, C0, 'k--', 'Separatriz')


t: list = np.linspace(0, -10, 1000)
inf = 0.01
plot_separatrix(t, inf, T_estacionario[0], C_estacionario[0])

t = np.linspace(0, 20, 1000)
T_initial = [330, 360]
C_initial = [9, 1]

plot_trajetories(t, T_initial[0], C_initial[0], 'b-', 'Simulação Dinâmica')

plot_trajetories(t, T_initial[1], C_initial[1], 'g-', 'Simulação Dinâmica')

plt.scatter(C_estacionario, T_estacionario, color='r',
            s=15, label='Estados Estacionários')
plt.scatter(C_initial, T_initial, color='c', s=15,
            marker='x', label='Condição Inicial')

plt.legend()
plt.ylim(300, 400)
plt.xlim(0, 10)
plt.xlabel('C [mol/L]')
plt.ylabel('T [K]')
plt.title('Plano de Fases de CxT')
plt.show()


# Número 2
# Ele quer que eu resolva ou avalie a estabilidade?
''' Aqui resolvi, mas arbitrando condição inicial
def dXdt(X, t):
    x1, x2 = X
    dx1_dt = x2
    dx2_dt = -x1 - u*x2
    return [dx1_dt, dx2_dt]


X0 = [10, 50]
u = 0
sol = odeint(dXdt, X0, t)

x1_plot = sol[:, 0]
x2_plot = sol[:, 1]

plt.plot(sol)
plt.show()
'''
'''
Aqui eu verifiquei a estabilidade
'''
