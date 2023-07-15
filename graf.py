import matplotlib.pyplot as plt
import numpy as np

# Dados para as curvas
x = np.linspace(0, 2*np.pi, 100)  # Valores de x de 0 a 2*pi
y1 = np.sin(x)                   # Valores de y para a primeira curva (seno)
y2 = np.cos(x)                   # Valores de y para a segunda curva (cosseno)

# Plotar as curvas
plt.plot(x, y1, label='Seno')
plt.plot(x, y2, label='Cosseno')

# Adicionar rótulos e título ao gráfico
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Curvas Seno e Cosseno')

# Adicionar uma legenda
plt.legend()

# Exibir o gráfico
plt.show()
