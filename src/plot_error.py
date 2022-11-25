# import require modules
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
# defining our function
x = np.array([0.1,0.3,0.5,0.7,1])
#y = np.array([1638.52,1520.42,1354.22,2213.90,16.25])
#y_error = np.array([35.09,31.97,22.53,23.10,0.10])
y = np.array([12.82,20.67,18.45,17.22,-0.02])
y_error = np.array([0.12,0.06,0.12,0.04,0.00])


#x = np.array([10,50,100,300,500])
#y = np.array([1331.96,1432.58,1064.22,1519.40,1994.16])
#y_error = np.array([370.21,26.08,27.21,31.73,67.37])
#y = np.array([14.75,14.44,14.21,20.67,17.49])
#y_error = np.array([4.62,1.19,0.19,0.06,0.08])

# plotting our function and
# error bar
#fig.suptitle('Control Cost with $\mu$=0.3 ', fontsize=20)
#plt.xlabel(r'$\eta$', fontsize=18)
plt.xlabel(r'$\mu$', fontsize=18)
plt.ylabel('Distance Covered', fontsize=16)
#plt.ylabel('Control Cost', fontsize=16)
plt.plot(x, y)

plt.errorbar(x, y, yerr = y_error, fmt ='o')

#plt.savefig("dc_beta.png")
plt.savefig("dc_mu.pdf")
