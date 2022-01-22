# Let's make a graph!
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 8*np.pi, 1001)
x = (t/2/np.pi) * np.cos(t)
y = (t/2/np.pi) * np.sin(t)

plt.plot(x,y, 'r-', lw=2)
plt.axis('equal')
plt.show()
