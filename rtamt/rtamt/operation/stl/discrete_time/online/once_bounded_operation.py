import collections
from rtamt.operation.abstract_operation import AbstractOperation
import numpy as np
import math
#from scipy.special import erf
from math import erf
import gvars

class OnceBoundedOperation(AbstractOperation):
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end
        self.buffer = collections.deque(maxlen=(self.end + 1))

        for i in range(self.end + 1):
            val = - float("inf")
            self.buffer.append(val)

    def reset(self):
        self.buffer = collections.deque(maxlen=(self.end + 1))

        for i in range(self.end + 1):
            val = - float("inf")
            self.buffer.append(val)

    def update(self, sample):
        self.buffer.append(sample)
        out = -float("inf")
        flist = lambda y:[x for a in y for x in flist(a)] if type(y) is list else [y]
        for i in range(self.end-self.begin+1):
            print(flist(self.buffer[i]))
            out = max(out, self.buffer[i])
        return out

    def update(self, sample, typeop):
        flist = lambda y:[x for a in y for x in flist(a)] if type(y) is list else [y]
        out=np.array(flist(sample), dtype=np.float128)
        #out = np.array(out_sample)
        #print("out is :  "+str(out))
        #rint("in STL")
        #rob_type=8
        #1 - classical, 
        #2 - agm
        #3 - Vanrai
        #4 - lse
        #print(rob_type)
        if typeop=="AND":
            #print("^^^^^^^^^^^")
            if gvars.rob_type==1:
                #classical rob
                self.buffer.append(min(out)) 
            elif gvars.rob_type==2:
                #AGM Robustness
                #print("agm")
                if (out<0).any():
                    #self.buffer.append(np.mean(out))
                    tmp=out
                    tmp[tmp>0]=0
                    self.buffer.append(np.mean(tmp))
                else:
                    self.buffer.append(((out+1).prod()**(1.0/len(out)))-1)
            
            elif gvars.rob_type==3:
                #Softmax Robustness
                #(Assuming nu=1)
                nu=3
                rmin=np.min(out)
                if rmin==0:
                    rmin=0.0001
                if (rmin<0):
                    rr=(out-rmin)/rmin
                    rr = np.clip(rr, a_min = -10, a_max = 10)
                    reff=rmin*np.exp(rr)
                    tmp=np.sum(reff*np.exp(nu*rr))/(np.sum(np.exp(nu*rr))+0.0001)
                    self.buffer.append(tmp)
                else:
                    rr=(out-rmin)/rmin
                    rr = np.clip(rr, a_min = -10, a_max = 10)
                    tmp=np.sum(out*np.exp(-1*nu*rr))/(np.sum(np.exp(-1*nu*rr))+0.0001)
                    self.buffer.append(tmp)
            elif gvars.rob_type==4:    
                #log-sum-exp Robustness
                #(Assuming beta=1)
                beta=1
                tmp=-1*np.log(np.sum(np.exp(-1*beta*out)))/beta
                self.buffer.append(tmp)
            elif gvars.rob_type==5:
                # SSS approx
                #mu=0.3
                mu=0.3
                beta=300
                #diff=np.subtract(max(out),min(out))
                diff=(np.log(np.sum(np.exp(beta*out)))+np.log(np.sum(np.exp(-1*beta*out))))/beta
                #print(out,"  >> ",diff)
                if math.isinf(diff) or math.isnan(diff):
                    beta=10
                    diff=(np.log(np.sum(np.exp(beta*out)))+np.log(np.sum(np.exp(-1*beta*out))))/beta
                #print(out, " >>> ",diff)
                tmp=(np.sum(out)-diff*erf(mu*diff))/len(out)
                self.buffer.append(tmp)
            # elif rob_type==5:             
            #     tmp=(np.sum(out)-np.subtract(max(out),min(out)))/len(out)
            #     self.buffer.append(tmp)
            # # elif rob_type==6:          
            # #     tmp=np.mean(out)-np.std(out)
            # #     self.buffer.append(tmp)
            # elif rob_type==7:   
            #     tmp=(np.sum(out)-2*out.std())/len(out)
            #     self.buffer.append(tmp)
            
            # elif rob_type==9:
            #     # add-sub-smooth approx
            #     mu=0.1
            #     tmp=(np.sum(out)-np.sqrt(np.square(np.subtract(max(out),min(out)))-mu))/len(out)
            #     self.buffer.append(tmp)
        elif typeop=="OR":
            raise NotImplementedError("Code for handling OR not added yet")
        else:
            self.buffer.append(out[0])

                

        #Impl code for OR         
        # else:
        #     #print("vvvvvvvvvvvvv")
        #     if (out>0).any():
        #         #print("neg")
        #         tmp=out
        #         tmp[tmp<0]=0
        #         self.buffer.append(np.mean(tmp))
        #     else:
        #         #print("pos")
        #         self.buffer.append(-1*(((1-out).prod()**(1.0/len(out)))+1))
          
        raw_trace = np.array(self.buffer)
        out_trace = np.array([v for v in raw_trace if not (math.isinf(v))],dtype=np.float128)
        #print("out_trace : "+str(out_trace))
        #print(self.begin)
        #print(self.end)
        #exit()
        if gvars.rob_type==1:
            #classical rob
            return max(out_trace) 
        elif gvars.rob_type==2:
            if (out_trace>0).any():
                tmp=out_trace
                tmp[tmp<0]=0
                return np.mean(tmp)
            else:
                return -1*((1-out_trace).prod()**(1.0/len(out_trace))+1)
        elif gvars.rob_type==3:
            #Softmax Robustness
            #(Assuming nu=1)
            nu=3
            rmax=np.max(out_trace)
            if rmax==0:
                rmax=0.0001
            if (rmax<0):
                rr=(rmax-out_trace)/rmax
                rr = np.clip(rr, a_min = -10, a_max = 10)
                reff=rmax*np.exp(-1*rr)
                tmp=np.sum(reff*np.exp(nu*rr))/(np.sum(np.exp(nu*rr))+0.0001)
                return tmp
            else:
                rr=(rmax-out_trace)/rmax
                rr = np.clip(rr, a_min = -10, a_max = 10)
                tmp=np.sum(out_trace*np.exp(-1*nu*rr))/(np.sum(np.exp(-1*nu*rr))+0.0001)
                return tmp
        elif gvars.rob_type==4:
            #log-sum-exp Robustness
            #(Assuming beta=1)
            beta=1
            tmp=np.log(np.sum(np.exp(beta*out_trace)))/beta
            return tmp
        elif gvars.rob_type==5:
            # SSS approx
            #mu=0.3
            mu=0.3
            beta=300
            #diff=np.subtract(max(out),min(out))
            diff=(np.log(np.sum(np.exp(beta*out_trace)))+np.log(np.sum(np.exp(-1*beta*out_trace))))/beta
            #print(out_trace,"  >> ",diff)
            if math.isinf(diff) or math.isnan(diff):
                    beta=10
                    diff=(np.log(np.sum(np.exp(beta*out_trace)))+np.log(np.sum(np.exp(-1*beta*out_trace))))/beta
            #print(out_trace, " >>> ",diff)
            tmp=(np.sum(out_trace)+diff*erf(mu*diff))/len(out_trace)
            return tmp

        # elif rob_type==5:            
        #     #log-sum-exp Robustness
        #     #(Assuming beta=1)
        #     tmp=(np.sum(out_trace)+np.subtract(max(out_trace),min(out_trace)))/len(out_trace)
        #     return tmp
        # # elif rob_type==6:            
        # #     tmp=np.mean(out)+np.std(out)
        # #     return tmp
        # elif rob_type==7:   
        #     tmp=(np.sum(out)+2*out.std())/len(out)
        #     return tmp
        
        # elif rob_type==9:
        #     # add-sub-smooth approx
        #     mu=0.1
        #     tmp=(np.sum(out_trace)+np.sqrt(np.square(np.subtract(max(out_trace),min(out_trace)))+mu))/len(out_trace)
        #     return tmp
            
                

            
        '''
        #print("update >>>>>>>>>>>>>>>>> "+str(sample))
        #self.buffer.append(sample)
        out = -float("inf")
        #print(out)
        
        print("buffer : "+str(self.buffer))
        #return flist(self.buffer[self.end])
        
        for i in range(self.end-self.begin+1):
            out = max(out, max(flist(self.buffer[i])))
        return out
        '''
