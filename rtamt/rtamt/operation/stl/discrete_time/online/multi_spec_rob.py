import numpy as np
import math
#from scipy.special import erf
from math import erf

def multi_spec_rob(rob):
    exit()
    #print(" multi spec rob")
    out=rob
    rob_type=2
        #1 - classical, 
        #2 - agm
        #3 - Vanrai
        #4 - lse
        #8 - sss
        #print(rob_type

    if rob_type==1:
        #classical rob
        return min(out)
    elif rob_type==2:
        #AGM Robustness
        if (out<0).any():
            #self.buffer.append(np.mean(out))
            tmp=out
            tmp[tmp>0]=0
            return np.mean(tmp)
        else:
            return (((out+1).prod()**(1.0/len(out)))-1)            
    elif rob_type==3:
        #Vanrai Robustness
        #(Assuming nu=1)
        nu=1
        rmin=np.min(out)
        if rmin==0:
            rmin=0.0001
        if (rmin<0):
            rr=(out-rmin)/rmin
            rr = np.clip(rr, a_min = -10, a_max = 10)
            reff=rmin*np.exp(rr)
            tmp=np.sum(reff*np.exp(nu*rr))/(np.sum(np.exp(nu*rr))+0.0001)
            return tmp
        else:
            rr=(out-rmin)/rmin
            rr = np.clip(rr, a_min = -10, a_max = 10)
            tmp=np.sum(out*np.exp(-1*nu*rr))/(np.sum(np.exp(-1*nu*rr))+0.0001)
            return tmp
    elif rob_type==4:    
        #log-sum-exp Robustness
        #(Assuming beta=1)
        beta=1
        tmp=-1*np.log(np.sum(np.exp(-1*beta*out)))/beta
        return tmp
    elif rob_type==5: 
        # add-sub approx           
        tmp=(np.sum(out)-np.subtract(max(out),min(out)))/len(out)
        return tmp
    elif rob_type==8:
        # erf approx
        mu=10
        tmp=(np.sum(out)-np.subtract(max(out),min(out))*erf(mu*np.subtract(max(out),min(out))))/len(out)
        return tmp
    elif rob_type==9:
        # add-sub-smooth approx
        mu=0.1
        tmp=(np.sum(out)-np.sqrt(np.square(np.subtract(max(out),min(out)))-mu))/len(out)
        return tmp
