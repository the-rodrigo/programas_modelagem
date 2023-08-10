import numpy as np


def jacobian(C, T, q, v, k0, deltaH_neg, E, Cp, hA, R):
    J = np.zeros((2, 2))

    J[0, 0] = -q/v-k0*np.exp(-E/(R*T))
    J[0, 1] = -k0*np.exp(-E/(R*T))*(E/(R*T*T))*C
    J[1, 0] = (deltaH_neg/Cp)*k0*np.exp(-E/(R*T))
    J[1, 1] = -q/v+(deltaH_neg/Cp)*k0*np.exp(-E/(R*T)) * \
        (E/(R*T*T))*C-(hA/(v*Cp))

    return J


if __name__ == '__main__':
    q = 0.1  # m^3/h
    v = 0.1  # m^3
    k0 = 9703*3600  # h^-1
    deltaH_neg = 5960  # kcal/kgmol
    E = 11843  # kcal/kgmol
    Cp = 500  # kcal/m^3K
    hA = 15  # kcal/hK
    R = 1.987  # kcal/molK
    Tc = 298.5  # K
    Te = 298.15  # K
    Ce = 10  # kgmol/m^3

    C = 5.68658739223607
    T = 337.7814448342708

    # Jacobian

    J = jacobian(C, T, q, v, k0, deltaH_neg, E, Cp, hA, R)

    # Eig values and vectors
    eig_val, eig_vec = np.linalg.eig(J)

    print(eig_val)
    print('')
    print(eig_vec)
