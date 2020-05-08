epslon = 1e-8
def F1(T):
    #create elson to prevent divide by zero

    recall = T[0]/(T[1] + epslon)
    precision = T[0]/(T[2] + epslon)
    return 2*(recall*precision)/(recall+precision + epslon), recall, precision

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __call__(self):
        return self.avg

def accuracy(outputs, labels):
	import numpy as np
	best_acc_diab = 0
	best_acc_glau = 0
	best_diabF1 = (0,0,0)
	best_glauF1 = (0,0,0)
	batchsize = len(labels[0])
	best_cutoff = 0
	for threshold in np.linspace(0, 1, 101):
		output = np.reshape(np.array(outputs),(2,-1))> threshold
		

		acc_diab = 0
		acc_diab = np.sum(output[0]==labels[0],dtype = float)
		acc_diab = acc_diab/float(batchsize)

		acc_glau = 0
		acc_glau = np.sum(output[1]==labels[1],dtype = float)
		acc_glau = acc_glau/float(batchsize)

		True_pos_diab = np.sum((output[0] == 1) & (labels[0] == 1))
		pos_diab = np.sum(1==labels[0],dtype = float)
		pos_redict_diab = np.sum(1==output[0],dtype = float)


		True_pos_glau = np.sum((output[1] == 1) & (labels[1] == 1),dtype = float)
		pos_glau = np.sum(labels[1] == 1,dtype = float)
		pos_redict_glau = np.sum(output[1] == 1,dtype = float)

		diabF1 = F1((True_pos_diab, pos_diab, pos_redict_diab))
		glauF1 = F1((True_pos_glau, pos_glau, pos_redict_glau))
		is_best = (2*(diabF1[0]*glauF1[0])/(diabF1[0]+glauF1[0] + epslon)) > (2*(best_diabF1[0]*best_glauF1[0])/(best_diabF1[0]+best_glauF1[0] + epslon))
		if is_best:
			best_diabF1 = diabF1
			best_glauF1 = glauF1
			best_acc_diab = acc_diab
			best_acc_glau = acc_glau
			best_cutoff = threshold

	return (best_acc_diab, best_acc_glau), best_diabF1, best_glauF1, best_cutoff
