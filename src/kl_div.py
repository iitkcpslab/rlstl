# box =[P(green),P(blue),P(red),P(yellow)]
box_1 = [0.25, 0.33, 0.23, 0.19]
box_2 = [0.21, 0.21, 0.32, 0.26]

import numpy as np
from scipy.special import rel_entr

def kl_divergence(a, b):
	return sum(a[i] * np.log(a[i]/b[i]) for i in range(len(a)))
	
print('KL-divergence(box_1 || box_2): %.3f ' % kl_divergence(box_1,box_2))
print('KL-divergence(box_2 || box_1): %.3f ' % kl_divergence(box_2,box_1))

# D( p || p) =0
print('KL-divergence(box_1 || box_1): %.3f ' % kl_divergence(box_1,box_1))

print("Using Scipy rel_entr function")
box_1 = np.array(box_1)
box_2 = np.array(box_2)

print('KL-divergence(box_1 || box_2): %.3f ' % sum(rel_entr(box_1,box_2)))
print('KL-divergence(box_2 || box_1): %.3f ' % sum(rel_entr(box_2,box_1)))
print('KL-divergence(box_1 || box_1): %.3f ' % sum(rel_entr(box_1,box_1)))

