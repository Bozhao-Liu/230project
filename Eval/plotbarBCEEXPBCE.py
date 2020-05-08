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
data = np.array([0.9391304296196599, 0.965517236212842, 0.9649122755340105, 0.9473684158879655, 0.9357798113626798]) - bottom

ax[0,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,0].set_title('F1 Score')
ax[0,0].set_ylabel('Diabetic Retinopathy')
ax[0,0].set_xticklabels(netlist, fontsize=10)


	#Recall
data = np.array([0.9310344825980975 , 0.9655172412128419, 0.9482758619054696, 0.9310344825980975, 0.879310344675981]) - bottom

ax[0,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,1].set_title('Recall')
ax[0,1].set_xticklabels(netlist, fontsize=10)

	#Precision
data = np.array([0.9473684208864266 , 0.9655172412128419, 0.9821428569674745, 0.9642857141135204, 0.9999999998039215]) - bottom

ax[0,2].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,2].set_title('Precision')
ax[0,2].set_xticklabels(netlist, fontsize=10)


#Glaucoma
	#F1 score
data = np.array([0.9535864928120494 , 0.9711934105580113, 0.961702122582164, 0.9749999949190975, 0.9527896944964911]) - bottom

ax[1,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,0].set_ylabel('Glaucoma')
ax[1,0].set_xticklabels(netlist, fontsize=10)

	#Recall
data = np.array([0.9338842974434807 , 0.9752066114896524, 0.9338842974434807, 0.9669421486804181, 0.917355371825012]) - bottom

ax[1,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,1].set_xlabel('Decay Rate')
ax[1,1].set_xticklabels(netlist, fontsize=10)

	#Precision
data = np.array([0.9741379309505054 , 0.9672131146748186, 0.9912280700884888, 0.9831932772283032, 0.99107142848294]) - bottom

ax[1,2].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,2].set_xticklabels(netlist, fontsize=10)

fig.suptitle('BCE balanced EXP_BCE learning rate decay comparison')
plt.savefig('BCEEXPBCEEval.png')
