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
data = np.array([0.909090903940496, 0.9391304296196599]) - bottom

ax[0,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,0].set_title('Diabetic Retinopathy')
ax[0,0].set_ylabel('F1')
ax[0,0].set_xticklabels(netlist, fontsize=10)

	#Recall
data = np.array([0.8620689653686088 , 0.9310344825980975]) - bottom

ax[1,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,0].set_ylabel('Recall')
ax[1,0].set_xticklabels(netlist, fontsize=10)

	#Precision
data = np.array([0.9615384613535503 , 0.9473684208864266]) - bottom
ax[2,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[2,0].set_ylabel('Precision')
ax[2,0].set_xticklabels(netlist, fontsize=10)

#Glaucoma
	#F1 score
data = np.array([0.9256198296342464 , 0.9535864928120494]) - bottom

ax[0,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,1].set_title('Glaucoma')
ax[0,1].set_xticklabels(netlist, fontsize=10)

	#Recall
data = np.array([0.9256198346342464 , 0.9338842974434807]) - bottom

ax[1,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,1].set_xticklabels(netlist, fontsize=10)

	#Precision
data = np.array([0.9256198346342464 , 0.9741379309505054]) - bottom
ax[2,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[2,1].set_xticklabels(netlist, fontsize=10)

fig.suptitle('BCE balanced EXPBCE: 1 LrD')
plt.savefig('BCEEXPBCE1LrD.png')
