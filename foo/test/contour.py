import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

x = np.array([1, 2])
y = np.array([1, 2, 3, 4])
z = np.array([[1, 2], [2, 3], [1, 5], [4, 5]])
# limit x, y
plt.xlim(1, 2)
plt.ylim(1, 5)
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:len(np.unique(z))])
plt.contourf(x, y, z, cmap=cmap)
plt.show()
