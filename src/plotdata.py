import pickle
import matplotlib.pyplot as plt
import csv
import numpy as np
p=[]
z=[]
a=[]
with open('log_hop.csv', mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            print(lines)     
            p.append(float(lines[1])-0.5)
            z.append(float(lines[2])-0.7)
            a.append(1-abs(float(lines[3])))
#plt.plot(p[1:200:5], label = "p-1",color="purple")
#plt.plot(z[1:200:5], label = "z-0.7",color="orange")
#plt.plot(a[1:200:5], label = "1-|a|",color="gray")
plt.plot(p[1:200:1], label = "p-0.5",color="purple")
plt.plot(z[1:200:1], label = "z-0.7",color="orange")
plt.plot(a[1:200:1], label = "1-|a|",color="gray")
maxi=np.maximum.reduce([p[1:200:1],z[1:200:1],a[1:200:1]])
mini=np.minimum.reduce([p[1:200:1],z[1:200:1],a[1:200:1]])
plt.plot(maxi-mini, label = "$\delta^{max}$",color="green")

#print(np.mean(p[1:25]))
#print(np.mean(z[1:25]))
#print(np.mean(a[1:25]))
print(p[0])
print(z[0])
print(a[0])
'''
x=[]
v=[]
with open('log_hc.csv', mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            print(lines)     
            x.append(float(lines[1]))
            v.append(float(lines[2]))
#plt.plot(x[1::5], label = "x",color="purple")
#plt.plot(v[1::5], label = "v",color="orange")
plt.plot(x[1:200:5], label = "x",color="purple")
plt.plot(v[1:200:5], label = "v",color="orange")
'''
plt.legend()
plt.xlabel('Timesteps', fontsize=12)
plt.ylabel('Robustness', fontsize=12)
plt.savefig("exampleHopData.png")
plt.show()


