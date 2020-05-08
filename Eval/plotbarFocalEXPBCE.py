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
data = np.array([0.955752207223745, 0.946428566265944, 0.946428566265944, 0.9572649521016876, 0.9473684158879655]) - bottom

ax[0,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,0].set_title('F1 Score')
ax[0,0].set_ylabel('Diabetic Retinopathy')
ax[0,0].set_xticklabels(netlist, fontsize=10)


	#Recall
data = np.array([0.9310344825980975 , 0.9137931032907253, 0.9137931032907253, 0.9655172412128419, 0.9310344825980975]) - bottom

ax[0,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,1].set_title('Recall')
ax[0,1].set_xticklabels(netlist, fontsize=10)

	#Precision
data = np.array([0.9818181816396694 , 0.9814814812997257, 0.9814814812997257, 0.9491525422120081, 0.9642857141135204]) - bottom

ax[0,2].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,2].set_title('Precision')
ax[0,2].set_xticklabels(netlist, fontsize=10)


#Glaucoma
	#F1 score
data = np.array([0.9586776808711838 , 0.9704641299414268, 0.9311740839954763, 0.9749999949190975, 0.9749999949190975]) - bottom

ax[1,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,0].set_ylabel('Glaucoma')
ax[1,0].set_xticklabels(netlist, fontsize=10)

	#Recall
data = np.array([0.9586776858711837 , 0.9504132230619494, 0.9504132230619494, 0.9669421486804181, 0.9669421486804181]) - bottom

ax[1,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,1].set_xlabel('Decay Rate')
ax[1,1].set_xticklabels(netlist, fontsize=10)

	#Precision
data = np.array([0.9586776858711837 , 0.9913793102593639, 0.9126984126259764, 0.9831932772283032, 0.9831932772283032]) - bottom

ax[1,2].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,2].set_xticklabels(netlist, fontsize=10)

fig.suptitle('Focal balanced EXP_BCE learning rate decay comparison')
plt.savefig('FocalEXPBCEEval.png')
