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
data = np.array([0.9107142805580358, 0.955752207223745]) - bottom

ax[0,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,0].set_title('Diabetic Retinopathy')
ax[0,0].set_ylabel('F1')
ax[0,0].set_xticklabels(netlist, fontsize=10)

	#Recall
data = np.array([0.879310344675981 , 0.9310344825980975]) - bottom

ax[1,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,0].set_ylabel('Recall')
ax[1,0].set_xticklabels(netlist, fontsize=10)

	#Precision
data = np.array([0.9444444442695473 , 0.9818181816396694]) - bottom
ax[2,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[2,0].set_ylabel('Precision')
ax[2,0].set_xticklabels(netlist, fontsize=10)

#Glaucoma
	#F1 score
data = np.array([0.936708855682672 , 0.9586776808711838]) - bottom

ax[0,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,1].set_title('Glaucoma')
ax[0,1].set_xticklabels(netlist, fontsize=10)

	#Recall
data = np.array([0.917355371825012 , 0.9586776858711837]) - bottom

ax[1,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,1].set_xticklabels(netlist, fontsize=10)

	#Precision
data = np.array([0.9568965516416469 , 0.9586776858711837]) - bottom
ax[2,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[2,1].set_xticklabels(netlist, fontsize=10)

fig.suptitle('Focal balanced EXP_BCE: 1 LrD')
plt.savefig('FocalEXPBCE1LrD.png')
