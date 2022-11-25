import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')

#fig = plt.figure(figsize = (10,10))
#ax = plt.axes(projection='3d')
#plt.show()

'''
x = np.array([10,15,20,25])
y = np.array([10,15,20,25])
z = np.array([1,2])

fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.grid()

ax.scatter(x, y, z, c = 'r', s = 50)
ax.set_title('3D Scatter Plot')

# Set axes label
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)

plt.show()
'''
x = [10, 15, 20, 25]
y = [10, 15, 20, 25]

X, Y = np.meshgrid(x, y)
print(X)
print(Y)
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')

#x = np.arange(-5, 5.1, 0.2)
#y = np.arange(-5, 5.1, 0.2)

X, Y = np.meshgrid(x, y)
Z = np.sin(X)*np.cos(Y)
#Z = [[4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]]
Z = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [8, 9, 2, 6], [2, 5, 6, 9]], np.int32)
print(type(Z))
surf = ax.plot_surface(X, Y, Z, cmap = plt.cm.cividis)

# Set axes label
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)

fig.colorbar(surf, shrink=0.5, aspect=8)

plt.show()
