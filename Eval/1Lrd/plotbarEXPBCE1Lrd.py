import numpy as np
import matplotlib.pyplot as plt

x = range(2)
x = np.array(x)+1
bottom = 0.85
width = 1
color = ['b', 'r']

fig, ax = plt.subplots(nrows=3, ncols=2)
netlist = ['', '', 'ResNet', '', 'DenseNet']


#Diabetic Retinopathy
	#F1 score
data = np.array([0.9217391252748582, 0.9482758569054697]) - bottom

ax[0,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,0].set_title('Diabetic Retinopathy')
ax[0,0].set_ylabel('F1')
ax[0,0].set_xticklabels(netlist, fontsize=10)

	#Recall
data = np.array([0.9137931032907253 , 0.9482758619054696]) - bottom

ax[1,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,0].set_ylabel('Recall')
ax[1,0].set_xticklabels(netlist, fontsize=10)

	#Precision
data = np.array([0.9298245612403816 , 0.9482758619054696]) - bottom
ax[2,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[2,0].set_ylabel('Precision')
ax[2,0].set_xticklabels(netlist, fontsize=10)

#Glaucoma
	#F1 score
data = np.array([0.9306122398227406 , 0.928571423505606]) - bottom

ax[0,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,1].set_title('Glaucoma')
ax[0,1].set_xticklabels(netlist, fontsize=10)

	#Recall
data = np.array([0.942148760252715 , 0.9669421486804181]) - bottom

ax[1,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,1].set_xticklabels(netlist, fontsize=10)

	#Precision
data = np.array([0.9193548386355359 , 0.8931297709241885]) - bottom
ax[2,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[2,1].set_xticklabels(netlist, fontsize=10)

fig.suptitle('EXP_BCE: 1 LrD')
plt.savefig('EXPBCE1LrD.png')
