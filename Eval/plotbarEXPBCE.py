import numpy as np
import matplotlib.pyplot as plt


x = range(5)
x = np.array(x)+1
bottom = 0.85
width = 1
color = ['b', 'r']

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,6))
netlist = ['', '1.00', '0.99', '0.98', '0.97', '0.96']


#Diabetic Retinopathy
	#F1 score
data = np.array([0.9482758569054697, 0.9369369317782649, 0.9482758569054697, 0.955752207223745, 0.9298245562419205]) - bottom

ax[0,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,0].set_title('F1 Score')
ax[0,0].set_ylabel('Diabetic Retinopathy')
ax[0,0].set_xticklabels(netlist, fontsize=10)


	#Recall
data = np.array([0.9482758619054696 , 0.8965517239833531, 0.9482758619054696, 0.9310344825980975, 0.9137931032907253]) - bottom

ax[0,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,1].set_title('Recall')
ax[0,1].set_xticklabels(netlist, fontsize=10)

	#Precision
data = np.array([0.9482758619054696 , 0.9811320752865789, 0.9482758619054696, 0.9818181816396694, 0.9464285712595664]) - bottom

ax[0,2].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,2].set_title('Precision')
ax[0,2].set_xticklabels(netlist, fontsize=10)


#Glaucoma
	#F1 score
data = np.array([0.928571423505606 , 0.9672131096751546, 0.9629629578837915, 0.9749999949190975, 0.9491525372956767]) - bottom

ax[1,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,0].set_ylabel('Glaucoma')
ax[1,0].set_xticklabels(netlist, fontsize=10)

	#Recall
data = np.array([0.9669421486804181 , 0.9752066114896524, 0.9669421486804181, 0.9669421486804181, 0.9256198346342464]) - bottom

ax[1,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,1].set_xlabel('Decay Rate')
ax[1,1].set_xticklabels(netlist, fontsize=10)

	#Precision
data = np.array([0.8931297709241885 , 0.9593495934179391, 0.9590163933640151, 0.9831932772283032, 0.9739130433935729]) - bottom

ax[1,2].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,2].set_xticklabels(netlist, fontsize=10)

fig.suptitle('EXP_BCE learning rate decay comparison')
plt.savefig('EXPBCEEval.png')
