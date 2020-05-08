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
data = np.array([0.9380530921826298, 0.9649122755340105, 0.955752207223745, 0.9380530921826298, 0.9369369317782649]) - bottom

ax[0,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,0].set_title('F1 Score')
ax[0,0].set_ylabel('Diabetic Retinopathy')
ax[0,0].set_xticklabels(netlist, fontsize=10)


	#Recall
data = np.array([0.9137931032907253 , 0.9482758619054696, 0.9310344825980975, 0.9137931032907253, 0.8965517239833531]) - bottom

ax[0,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,1].set_title('Recall')
ax[0,1].set_xticklabels(netlist, fontsize=10)

	#Precision
data = np.array([0.963636363461157 , 0.9821428569674745, 0.9818181816396694, 0.963636363461157, 0.9811320752865789]) - bottom

ax[0,2].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,2].set_title('Precision')
ax[0,2].set_xticklabels(netlist, fontsize=10)


#Glaucoma
	#F1 score
data = np.array([0.9790794928268064 , 0.9704641299414268, 0.9749999949190975, 0.9661016898366132, 0.9583333282538196]) - bottom

ax[1,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,0].set_ylabel('Glaucoma')
ax[1,0].set_xticklabels(netlist, fontsize=10)

	#Recall
data = np.array([0.9669421486804181 , 0.9504132230619494, 0.9669421486804181, 0.942148760252715, 0.9504132230619494]) - bottom

ax[1,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,1].set_xlabel('Decay Rate')
ax[1,1].set_xticklabels(netlist, fontsize=10)

	#Precision
data = np.array([0.991525423644786 , 0.9913793102593639, 0.9831932772283032, 0.9913043477398866, 0.9663865545406398]) - bottom

ax[1,2].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,2].set_xticklabels(netlist, fontsize=10)

fig.suptitle('Focal loss learning rate decay comparison')
plt.savefig('FocalEval.png')
