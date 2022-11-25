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

x1 = [1,5,5,2]
x2 = [1,3,4,5]
x3 = [2,6,5,4]
x4 = [3,7,6,4]

rob1 = []
for x in [x1,x2,x3,x4]:
    out=np.array(x)
    rob1.append(min(out))
    #print(out)
#rob1 = np.minimum(xx,y)
print("classical ", rob1)

rob2 = []
for x in [x1,x2,x3,x4]:
    out=np.array(x)
    if (out<0).any():
        tmp=out
        tmp[tmp>0]=0
        rob2.append(np.mean(tmp))
    else:
        rob2.append(((out+1).prod()**(1.0/len(out)))-1)
    #rob2.append(((out+1).prod()**(1.0/len(out)))-1)
print("agm ", rob2)


rob31 = []
for x in [x1,x2,x3,x4]:
    out=np.array(x)
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
print("varnai ", rob31)

rob32 = []
for x in [x1,x2,x3,x4]:
    out=np.array(x)
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
for x in [x1,x2,x3,x4]:
    out=np.array(x)
    beta=1
    rob41.append(-1*np.log(np.sum(np.exp(-1*beta*out)))/beta)
print("lse ",rob41)


rob42 = []
for x in [x1,x2,x3,x4]:
    out=np.array(x)
    beta=100
    tmp=np.exp(-1*beta*out)
    #tmp = np.clip(np.exp(-1*beta*out), a_min = -1e+304, a_max = 1e+304)
    #print("tmp :"+str(tmp))
    rob42.append(-1*np.log(np.sum(tmp))/beta)
#print(rob42)


#from scipy.special import erf
from math import erf
rob51 = []
for x in [x1,x2,x3,x4]:
    out=np.array(x)
    #mu=0.3
    mu=1.2
    tmp=(np.sum(out)-np.subtract(max(out),min(out))*erf(mu*np.subtract(max(out),min(out))))/len(out)
    #tmp = np.clip(np.exp(-1*beta*out), a_min = -1e+304, a_max = 1e+304)
    #print("tmp :"+str(tmp))
    rob51.append(tmp)
print("sss ", rob51)

rob52 = []
for x in [x1,x2,x3,x4]:
    out=np.array(x)
    tmp=(np.sum(out)-2*out.std())/len(out)
    #print("tmp :"+str(tmp))
    rob52.append(tmp)
#print(rob53)




'''
import matplotlib.pyplot as plt
  
# plot lines
plt.plot(xx, rob1, label = "classical",color="red")
plt.plot(xx, rob2, label = "agm",color="magenta")
plt.plot(xx, rob31, label = "softmax",color="orange")
#plt.plot(xx, rob32, label = "v100",color="green")
#plt.plot(xx, rob41, label = "lse",color="brown")
#plt.plot(xx, rob42, label = "l100",color="blue")
plt.plot(xx, rob51, label = "sss",color="black")
plt.plot(xx, rob52, label = "ssd",color="green")
plt.legend()
plt.xlabel('x2', fontsize=12)
plt.ylabel('robustness of AND(x1,x2)', fontsize=12)
plt.savefig("compare.png")
plt.show()
'''
