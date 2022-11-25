# RLSTL   
## RL based synthesis of feedback controller from STL specifications   

RLSTL is a tool to synthesize feedback controllers using Signal Temporal Logic (STL). It is developed using the stable-baselines3 framework (https://github.com/DLR-RM/stable-baselines3) and rtamt online monitoring tool (https://github.com/nickovic/rtamt).


## Main Features

- Updated SAC algorithm to use online STL robustness.  
- Implemented some of the State-of-the-art semantics of STL to RTAMT tool.  
- Developed new STL semantics and added it to RTAMT tool.
- Added script for evaluation of different Controllers.    
  

### Prerequisites

Use Ubuntu 20.04 and above.  
Packages: libboost-all-dev, python-dev, python-pip, antlr4   

Install mujoco (https://github.com/openai/mujoco-py).   

Python packages    
<Pkg>        <Preferable Version>   
Python         3.7     
torch          1.11.0      
gym            0.21.0     
PyOpenGL       3.1.5    
glfw           2.4.0   
imageio        2.10.3   
mujoco-py      >=2.1.2   

The above version are highly recommended due issues with other versions.   
For instance, mujoco==2.1.2.12 conflicts with gym 0.24.0.   

## Installation
Create a python3.7 virtual environment and do the following:   

Unzip the file RLSTL.zip and install the RLSTL package:    
```
cd RLSTL/
pip install -e .
```

Install the rtamt package   
```
cd rtamt/
pip3 install .
```

After installation add the path to the "RLSTL" and the "RLSTL/rtamt" package   
to the PYTHONPATH variable in ~/.bashrc file.    
For example, if the RLSTL package is ta location /home/PC/RLSTL/, then add the  
following lines in the ~/.bashrc file:  
export PYTHONPATH=/home/PC/RLSTL/:$PYTHONPATH                       
export PYTHONPATH=/home/PC/RLSTL/rtamt/:$PYTHONPATH  

Alternatively, instead to adding these lines to ~/.bashrc, you can run these  
two lines in the terminal but it will be valid for a session only.   


(In case this variable is not set properly, you might notice error saying  
"TypeError: learn() got an unexpected keyword argument 'reward_type').   



## Running Experiments  

### Synthesizing Controllers    

```
cd src/
```

```
python run_experiments.py --env=<Env> --sem=<semantics-id> --run=<run-id>
```
where Env={HalfCheetah-v3,Hopper-v3,Ant-v3,Walker2d-v3,Swimmer-v3,Humanoid-v3}   
semantics-id is an integer in range [1,6]   

id  Semantics Name       
1 - Classical (cls)   
2 - AGM (agm)   
3 - LSE (lse)   
4 - Softmax (smax)   
5 - SSS (sss)   
   
run-id is a unique integer to be provided by the user. This purpose is   
to distinguish one set of experiments from the other.   

For example, to synthesize the controller for Hopper with semantics SSS 
for the first time, run the command:    

```
python run_experiments.py --env=Hopper-v3 --sem=5 --run=1
```

Once the experiment finishes, it will create a controller   
file named sac_Env_semantics-id_run-id.zip.   
Let us refer to this controller as C.      

Alternatively, you can copy the controller files from the   
src/controller/ folder.          

### Evaluation  

For evaluation of a controller for environemnt <Env> 
rename the file C to sac_Env.zip 

```
python evaluator.py <Env>
```
For example to evaluate the controller for Hopper,   
Rename the file to sac_Hopper-v3.zip (i.e. trim the   
semantics-id and runid)   
Then run the command: python evaluator.py Hopper-v3

For evaluation (alternatively), you can copy the controller   
files to src/ folder and run evaluator.py.       
For example, to evaluate the controller for SSS semantics for   
Hopper benchmark, goto the src/ folder, do:   

```
cp controllers/hopper/sac_Hopper-v3_sss_17.zip sac_Hopper-v3.zip 
python evaluator.py Hopper-v3 
```



For checking safety
```
python check_safety.py <Env>
```
For example, to check safety of the controller for Hopper,   
Rename the file to sac_Hopper-v3.zip (i.e. trim the   
semantics-id and runid)   
Then run the command: python check_safety.py Hopper-v3


### Files
All the source code is inside the src/ folder.   
Inside src/ the controllers/ folder contains the controllers for  
each benchmark.  



### Reproducibilty
All Experiments have been performed on a PC with i7-4770 3.40 GHzX8 CPU,   
32GB RAM and Ubuntu 20.04 OS.   

Completely reproducible results are not guaranteed across PyTorch releases or different platforms.   
Refer to the following notes by   
PyTorch (https://pytorch.org/docs/stable/notes/randomness.html) and   
stable-baselines (https://stable-baselines3.readthedocs.io/en/master/guide/algos.html#reproducibility)    
