import numpy as np
import matplotlib.pyplot as plt

x = range(2)
x = np.array(x)+1
bottom = 0.85
width = 1
color = ['b', 'r']

fig, ax = plt.subplots(nrows=3, ncols=2)
netlist = ['', '', 'ResNet', '', 'DenseNet']

#F1 score
data = np.array([0.9026548621003995, 0.9380530921826298]) - bottom

ax[0,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,0].set_title('Diabetic Retinopathy')
ax[0,0].set_ylabel('F1')
ax[0,0].set_xticklabels(netlist, fontsize=10)

data = np.array([0.9499999949211806 , 0.9790794928268064]) - bottom

ax[0,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[0,1].set_title('Glaucoma')
ax[0,1].set_xticklabels(netlist, fontsize=10)


#Recall
	#Diabetic Retinopathy
data = np.array([0.879310344675981 , 0.9137931032907253]) - bottom

ax[1,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,0].set_ylabel('Recall')
ax[1,0].set_xticklabels(netlist, fontsize=10)
	#Glaucoma
data = np.array([0.942148760252715 , 0.9669421486804181]) - bottom

ax[1,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[1,1].set_xticklabels(netlist, fontsize=10)


#Precision
	#Diabetic Retinopathy
data = np.array([0.9272727271041322 , 0.963636363461157]) - bottom
ax[2,0].bar(x, data, width = width, bottom = bottom, color = color)
ax[2,0].set_ylabel('Precision')
ax[2,0].set_xticklabels(netlist, fontsize=10)
	#Glaucoma
data = np.array([0.9579831931968081 , 0.991525423644786]) - bottom
ax[2,1].bar(x, data, width = width, bottom = bottom, color = color)
ax[2,1].set_xticklabels(netlist, fontsize=10)

fig.suptitle('Focal Loss: 1 LrD')
plt.savefig('Focal1LrD.png')
