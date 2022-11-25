import numpy as np




# for neg
y=-5
xx = [-1,-2,-3,-4,-5,-6,-7,-8,-9]

#for pos
y=5
xx = [1,2,3,4,5,6,7,8,9]

# for testing  only
y=0.5
xx = np.arange(0,1,0.1) 
#y=5
#xx = [5,4,3,2,1]

rob1 = []
for x in xx:
    out=np.array([x,y])
    rob1.append(min(out))
    #print(out)
#rob1 = np.minimum(xx,y)
print(rob1)


rob2 = []
for x in xx:
    out=np.array([x,y])
    if (out<0).any():
        tmp=out
        tmp[tmp>0]=0
        rob2.append(np.mean(tmp))
    else:
        rob2.append(((out+1).prod()**(1.0/len(out)))-1)
    #rob2.append(((out+1).prod()**(1.0/len(out)))-1)
print(rob2)


rob31 = []
for x in xx:
    out=np.array([x,y])
    nu=1
    rmin=np.min(out)
    if rmin==0:
        rmin=0.0001
    if (rmin<0):
        rr=(out-rmin)/rmin
        rr = np.clip(rr, a_min = -10, a_max = 10)
        reff=rmin*np.exp(rr)
        tmp=np.sum(reff*np.exp(nu*rr))/(np.sum(np.exp(nu*rr))+0.0001)
        rob31.append(tmp)
    else:
        rr=(out-rmin)/rmin
        rr = np.clip(rr, a_min = -10, a_max = 10)
        tmp=np.sum(out*np.exp(-1*nu*rr))/(np.sum(np.exp(-1*nu*rr))+0.0001)
        rob31.append(tmp)
    #rr=(out-rmin)/rmin
    #rr = np.clip(rr, a_min = -10, a_max = 10)
    #rob31.append(np.sum(out*np.exp(-1*nu*rr))/(np.sum(np.exp(-1*nu*rr))+0.0001))
print(rob31)

rob32 = []
for x in xx:
    out=np.array([x,y])
    nu=100
    rmin=np.min(out)
    if rmin==0:
        rmin=0.0001
    if (rmin<0):
        rr=(out-rmin)/rmin
        rr = np.clip(rr, a_min = -10, a_max = 10)
        reff=rmin*np.exp(rr)
        tmp=np.sum(reff*np.exp(nu*rr))/(np.sum(np.exp(nu*rr))+0.0001)
        rob32.append(tmp)
    else:
        rr=(out-rmin)/rmin
        rr = np.clip(rr, a_min = -10, a_max = 10)
        tmp=np.sum(out*np.exp(-1*nu*rr))/(np.sum(np.exp(-1*nu*rr))+0.0001)
        rob32.append(tmp)
    #rr=(out-rmin)/rmin
    #rr = np.clip(rr, a_min = -10, a_max = 10)
    #rob32.append(np.sum(out*np.exp(-1*nu*rr))/(np.sum(np.exp(-1*nu*rr))))
#print(rob32)



rob41 = []
for x in xx:
    out=np.array([x,y])
    beta=1
    rob41.append(-1*np.log(np.sum(np.exp(-1*beta*out)))/beta)
print(rob41)


rob42 = []
for x in xx:
    out=np.array([x,y])
    beta=100
    tmp=np.exp(-1*beta*out)
    #tmp = np.clip(np.exp(-1*beta*out), a_min = -1e+304, a_max = 1e+304)
    #print("tmp :"+str(tmp))
    rob42.append(-1*np.log(np.sum(tmp))/beta)
#print(rob42)


rob51 = []
for x in xx:
    out=np.array([x,y])
    alpha=0.1
    tmp=np.quantile(out,alpha)
    #tmp = np.clip(np.exp(-1*beta*out), a_min = -1e+304, a_max = 1e+304)
    #print("tmp :"+str(tmp))
    rob51.append(tmp)
#print(rob51)

#from scipy.special import erf
from math import erf
rob52 = []
for x in xx:
    out=np.array([x,y])
    #mu=0.3
    #mu=0.15
    mu=1.5
    tmp=(np.sum(out)-np.subtract(max(out),min(out))*erf(mu*np.subtract(max(out),min(out))))/len(out)
    #tmp = np.clip(np.exp(-1*beta*out), a_min = -1e+304, a_max = 1e+304)
    #print("tmp :"+str(tmp))
    rob52.append(tmp)
print(rob52)

rob53 = []
for x in xx:
    out=np.array([x,y])
    tmp=(np.sum(out)-2*out.std())/len(out)
    #print("tmp :"+str(tmp))
    rob53.append(tmp)
#print(rob53)

rob54 = []
for x in xx:
    out=np.array([x,y])
    mu=1
    tmp=np.mean(out)-np.std(out)
    #tmp = np.clip(np.exp(-1*beta*out), a_min = -1e+304, a_max = 1e+304)
    #print("tmp :"+str(tmp))
    rob54.append(tmp)
#print(rob54)


rob55 = []
for x in xx:
    out=np.array([x,y])
    mu=1
    tmp=0.5*(np.sum(out)-np.subtract(max(out),min(out))*np.tanh(mu*np.subtract(max(out),min(out))))
    #tmp = np.clip(np.exp(-1*beta*out), a_min = -1e+304, a_max = 1e+304)
    #print("tmp :"+str(tmp))
    rob55.append(tmp)
#print(rob55)

rob56 = []
for x in xx:
    out=np.array([x,y])
    mu=1
    tmp=0.5*(np.sum(out)-np.subtract(max(out),min(out))*(2/np.pi)*np.arctan(np.sinh(mu*np.subtract(max(out),min(out)))))
    #tmp = np.clip(np.exp(-1*beta*out), a_min = -1e+304, a_max = 1e+304)
    #print("tmp :"+str(tmp))
    rob56.append(tmp)
#print(rob56)


rob57 = []
for x in xx:
    out=np.array([x,y])
    mu=0.1
    tmp=(np.sum(out)-(np.subtract(max(out),min(out))-mu))/len(out)
    #tmp = np.clip(np.exp(-1*beta*out), a_min = -1e+304, a_max = 1e+304)
    #print("tmp :"+str(tmp))
    rob57.append(tmp)
#print(rob57)

import matplotlib.pyplot as plt
  
# plot lines
plt.plot(xx, rob1, label = "classical",color="red")
plt.plot(xx, rob2, label = "agm",color="magenta")
plt.plot(xx, rob31, label = "softmax",color="orange")
#plt.plot(xx, rob32, label = "v100",color="green")
#plt.plot(xx, rob41, label = "lse",color="brown")
plt.plot(xx, rob42, label = "l100",color="blue")
#plt.plot(xx, rob51, label = "quantile",color="brown")
plt.plot(xx, rob52, label = "sss",color="black")
#plt.plot(xx, rob53, label = "ssd",color="green")
#plt.plot(xx, rob54, label = "mstd",color="black")
#plt.plot(xx, rob55, label = "tanh",color="green")
#plt.plot(xx, rob57, label = "add-sub-approx",color="brown")
plt.legend()
plt.xlabel('x2', fontsize=12)
plt.ylabel('robustness of AND(x1,x2)', fontsize=12)
plt.savefig("compare.png")
plt.show()

