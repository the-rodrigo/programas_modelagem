import math

import matplotlib.pyplot as plt
import numpy as np

import jacobian_CSTR as jac


def get_C(T):
    return q*Cf/(q + v*k0*math.exp(-E/(R*T)))


def get_newT(T, C, Tcf):
    return (deltaH_neg*v*k0*C*math.exp(-E/(R*T)) + hA*Tcf
            + q*Cp*Tf)/(q*Cp + hA)


q: float = 0.1
v: float = 0.1
k0: float = 9703*3600
deltaH_neg: float = 5960
E: float = 11843
Cp: float = 500
hA: float = 15
R: float = 1.987
Tf: float = 298.15
Cf: float = 10

n = 10000
Tcf_vector = np.linspace(300, 400, n)
Real_1 = np.zeros(n)
Imag_1 = np.zeros(n)
Real_2 = np.zeros(n)
Imag_2 = np.zeros(n)
J = []

for j, temperature in enumerate(Tcf_vector):
    tolerance = 10**-25
    error = 1
    T0 = 337
    C = []
    T = [T0]
    i = 0
    while error > tolerance and i <= 10000:
        C.append(get_C(T[i]))
        T.append(get_newT(T[i], C[i], temperature))
        error = np.absolute((T[i+1]-T[i])/T[i])
        i += 1

    TEE = T[-1]
    CEE = C[-1]

    J = jac.jacobian(CEE, TEE, q, v, k0, deltaH_neg, E, Cp, hA, R)

    eig_val = np.linalg.eig(J)[0]

    Real_1[j] = np.real(eig_val[0])
    Imag_1[j] = np.imag(eig_val[0])

    Real_2[j] = np.real(eig_val[1])
    Imag_2[j] = np.imag(eig_val[1])


plt.plot(Real_1, Imag_1, 'r-', Real_2, Imag_2, 'b-', linewidth=1)
plt.title('Root locus van der Pol')
plt.xlabel('Re(λ)')
plt.ylabel('Im(λ)')
plt.plot(Real_1[0], Imag_1[0], 'rx', Real_1[n-1],
         Imag_1[n-1], 'ro', linewidth=1, markersize=5)
plt.plot(Real_2[0], Imag_2[0], 'bx', Real_2[n-1],
         Imag_2[n-1], 'bo', linewidth=1, markersize=5)
plt.legend(['λ_1', 'λ_2', 'Begin', 'End', 'Begin', 'End'])
plt.show()
