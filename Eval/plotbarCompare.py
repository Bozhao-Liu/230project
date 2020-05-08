import numpy as np
import matplotlib.pyplot as plt


x = range(5)
x = np.array(x)+1
bottom = 0.92
width = 1
color = ['b', 'r']

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,6))
netlist = ['', 'BCE', 'EXP_BCE', 'Focal', 'F-EXP', 'B-EXP']


#Diabetic Retinopathy
	#F1 score
data = np.array([0.955752207223745, 0.9482758569054697, 0.9649122755340105, 0.9572649521016876, 0.965517236212842]) - bottom

ax[0,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,0].set_title('F1 Score')
ax[0,0].set_ylabel('Diabetic Retinopathy')
ax[0,0].set_xticklabels(netlist, fontsize=10)


	#Recall
data = np.array([0.9310344825980975 , 0.9482758619054696, 0.9482758619054696, 0.9655172412128419, 0.9655172412128419]) - bottom

ax[0,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,1].set_title('Recall')
ax[0,1].set_xticklabels(netlist, fontsize=10)

	#Precision
data = np.array([0.9818181816396694 , 0.9482758619054696, 0.9821428569674745, 0.9491525422120081, 0.9655172412128419]) - bottom

ax[0,2].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,2].set_title('Precision')
ax[0,2].set_xticklabels(netlist, fontsize=10)


#Glaucoma
	#F1 score
data = np.array([0.9749999949190975 , 0.9629629578837915, 0.9704641299414268, 0.9749999949190975, 0.9711934105580113]) - bottom

ax[1,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,0].set_ylabel('Glaucoma')
ax[1,0].set_xticklabels(netlist, fontsize=10)

	#Recall
data = np.array([0.9669421486804181 , 0.9669421486804181, 0.9504132230619494, 0.9669421486804181, 0.9752066114896524]) - bottom

ax[1,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,1].set_xlabel('Decay Rate')
ax[1,1].set_xticklabels(netlist, fontsize=10)

	#Precision
data = np.array([0.9831932772283032 , 0.9590163933640151, 0.9913793102593639, 0.9831932772283032, 0.9672131146748186]) - bottom

ax[1,2].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,2].set_xticklabels(netlist, fontsize=10)

fig.suptitle('Learning rate decay comparison')
plt.savefig('CampareEval.png')
