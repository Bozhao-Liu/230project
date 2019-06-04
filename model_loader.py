import os
import sys
import torch
import logging

def loadModel(netname, params, pretrained = True):
    Netpath = 'Net'
    Netfile = os.path.join(Netpath, netname + '.py')
    assert os.path.isfile(Netfile), "No python file found for {}, (file name is case sensitive)".format(Netfile)
    netname = netname.lower()
    if netname == 'alexnet': 
        model, version = loadAlexnet(pretrained)
    elif 'densenet': 
        model, version = loadDensenet(pretrained, params)
    else:
        logging.warning("No model with the name {} found, please check your spelling.".format(netname))
        logging.warning("Net List:")
        logging.warning("    AlexNet")
        logging.warning("    DenseNet")
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

def UnevenWeightBCE_loss(outputs, labels, weights = (1, 1)):
    '''
    Cross entropy loss with uneven weigth between positive and negative result to manually adjust precision and recall
    '''
    loss_diab = torch.sum(torch.add(weights[0]*torch.mul(labels[:, 0],torch.log(outputs[:, 0])), weights[1]*torch.mul(1 - labels[:, 0],torch.log(1 - outputs[:, 0]))))
    loss_glau = torch.sum(torch.add(weights[0]*torch.mul(labels[:, 1],torch.log(outputs[:, 1])), weights[1]*torch.mul(1 - labels[:, 1],torch.log(1 - outputs[:, 1]))))
    return -torch.add(loss_diab, loss_glau)

def Exp_UEW_BCE_loss(outputs, labels, weights = (1, 1)):
    '''
    Cross entropy loss with uneven weigth between positive and negative result, add exponential function to positive to manually adjust precision and recall
    '''
    loss_diab = torch.sum(torch.add(weights[0]*torch.exp(torch.mul(labels[:, 0],torch.log(outputs[:, 0])))-1, weights[1]*torch.mul(1 - labels[:, 0],torch.log(1 - outputs[:, 0]))))
    loss_glau = torch.sum(torch.add(weights[0]*torch.exp(torch.mul(labels[:, 1],torch.log(outputs[:, 1])))-1, weights[1]*torch.mul(1 - labels[:, 1],torch.log(1 - outputs[:, 1]))))
    return -torch.add(loss_diab, loss_glau)