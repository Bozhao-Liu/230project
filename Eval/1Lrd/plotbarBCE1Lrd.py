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
data = np.array([0.9203539771415147, 0.946428566265944]) - bottom

ax[0,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,0].set_title('Diabetic Retinopathy')
ax[0,0].set_ylabel('F1')
ax[0,0].set_xticklabels(netlist, fontsize=10)

	#Recall
data = np.array([0.8965517239833531 , 0.9137931032907253]) - bottom

ax[1,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,0].set_ylabel('Recall')
ax[1,0].set_xticklabels(netlist, fontsize=10)

	#Precision
data = np.array([0.9454545452826446 , 0.9814814812997257]) - bottom
ax[2,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[2,0].set_ylabel('Precision')
ax[2,0].set_xticklabels(netlist, fontsize=10)

#Glaucoma
	#F1 score
data = np.array([0.9416666615885418 , 0.9669421436804181]) - bottom

ax[0,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,1].set_title('Glaucoma')
ax[0,1].set_xticklabels(netlist, fontsize=10)

	#Recall
data = np.array([0.9338842974434807 , 0.9669421486804181]) - bottom

ax[1,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,1].set_xticklabels(netlist, fontsize=10)

	#Precision
data = np.array([0.9495798318529766 , 0.9669421486804181]) - bottom
ax[2,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[2,1].set_xticklabels(netlist, fontsize=10)

fig.suptitle('BCE: 1 LrD')
plt.savefig('BCE1LrD.png')
