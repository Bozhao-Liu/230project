import os
import sys
import torch
import logging

epslon = 1e-8
def loadModel(netname, params, pretrained = True):
    Netpath = 'Net'
    Netfile = os.path.join(Netpath, netname + '.py')
    assert os.path.isfile(Netfile), "No python file found for {}, (file name is case sensitive)".format(Netfile)
    netname = netname.lower()
    if netname == 'alexnet': 
        model, version = loadAlexnet(pretrained)
    elif netname == 'densenet': 
        model, version = loadDensenet(pretrained, params)
    elif netname == 'smallresnet': 
        model, version = loadSmallResNet()
    else:
        logging.warning("No model with the name {} found, please check your spelling.".format(netname))
        logging.warning("Net List:")
        logging.warning("    AlexNet")
        logging.warning("    DenseNet")
        logging.warning("    SmallResNet")
        sys.exit()
    return model, version
    
def loadAlexnet(pretrained):
    import Net.alexnet
    print("Loading AlexNet")
    return Net.alexnet.alexnet(pretrained = pretrained, num_classes = 2), ''
    
def loadDensenet(pretrained, params):
    import Net.densenet
    print("Loading DenseNet")
    return Net.densenet.net(str(params.version), pretrained), str(params.version)

def loadSmallResNet():
    import Net.smallresnet
    print("Loading SmallResNet")
    return Net.smallresnet.smallresnet(num_classes=2), ''

def UnevenWeightBCE_loss(outputs, labels, weights = (1, 1)):
    '''
    Cross entropy loss with uneven weigth between positive and negative result to manually adjust precision and recall
    '''
    loss = [torch.sum(torch.add(weights[0]*torch.mul(labels[:, i],torch.log(outputs[:, i])), weights[1]*torch.mul(1 - labels[:, i],torch.log(1 - outputs[:, i])))) for i in range(outputs.shape[1])]
    return -torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)

def Exp_UEW_BCE_loss(outputs, labels, weights = (1, 1)):
	'''
	Cross entropy loss with uneven weigth between positive and negative result, add exponential function to positive to manually adjust precision and recall
	'''
	loss = [torch.sum(torch.add(weights[0]*torch.mul(labels[:, i],1.0/(outputs[:, i]+epslon) - 1), weights[1]*torch.mul(1 - labels[:, i],1.0/(1 - outputs[:, i]+epslon)-1))) for i in range(outputs.shape[1])]
	return torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)

def Exp_UEW_BCE_loss_BCE_balanced(outputs, labels, weights = (1, 1)):
	'''
	Cross entropy loss with uneven weigth between positive and negative result, add exponential function to positive to manually adjust precision and recall
	'''
	loss = [torch.sum(torch.add(weights[0]*torch.mul(labels[:, i],1.0/(outputs[:, i]+epslon) - 1), -weights[1]*torch.mul(1 - labels[:, i],torch.log(1 - outputs[:, i]+epslon)))) for i in range(outputs.shape[1])]
	return torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)

def Exp_UEW_BCE_loss_focal_balanced(outputs, labels, gamma = 2):
	'''
	Cross entropy loss with uneven weigth between positive and negative result, add exponential function to positive to manually adjust precision and recall
	'''
	loss = [torch.sum(torch.add(	torch.mul(labels[:, i],1.0/(outputs[:, i]+epslon) - 1), 
					-torch.mul(torch.pow(outputs[:, i], gamma), torch.mul(1 - labels[:, i], torch.log(1 - outputs[:, i]+epslon))))) for i in range(outputs.shape[1])]
	return torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)

def Focal_loss(outputs, labels, gamma = (2, 2)):
	loss = [torch.sum(torch.add(	torch.mul(torch.pow(1 - outputs[:, i], gamma[0]), torch.mul(labels[:, i], torch.log(outputs[:, i]+epslon))), 
					torch.mul(torch.pow(outputs[:, i], gamma[1]), torch.mul(1 - labels[:, i], torch.log(1 - outputs[:, i]+epslon))))) for i in range(outputs.shape[1])]
	return -torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)

def get_loss(loss_name):
	loss_name = loss_name.lower()
	if loss_name == 'bce':
		return UnevenWeightBCE_loss
	elif loss_name == 'exp_bce': 
        	return Exp_UEW_BCE_loss
	elif loss_name == 'focal': 
        	return Focal_loss
	elif loss_name == 'expbce_focal_balanced': 
        	return Exp_UEW_BCE_loss_focal_balanced
	elif loss_name == 'expbce_bce_balanced': 
        	return Exp_UEW_BCE_loss_BCE_balanced
	else:
		logging.warning("No loss function with the name {} found, please check your spelling.".format(loss_name))
		logging.warning("loss function List:")
		logging.warning("    BCE")
		logging.warning("    EXP_BCE")
		logging.warning("    Focal")
		logging.warning("    EXPBCE_Focal_Balanced")
		logging.warning("    EXPBCE_BCE_Balanced")
		import sys
		sys.exit()
