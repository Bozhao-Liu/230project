"""Defines the neural network, losss function and metrics"""

import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import model.functions as functions

class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions

    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_channels = params.num_channels
        self.input_width = params.input_width
        self.input_height = params.input_height
        
        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        self.conv1 = nn.Conv2d(params.num_input_channels, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)

        # 2 fully connected layers to transform the output of the convolution layers
        self.fc1 = nn.Linear(int(self.input_width*self.input_height/64)*self.num_channels*4, self.num_channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)    
        self.dropout_rate = params.dropout_rate

        # 3 fully connected layers to transform the output of the convolution layers to the multi-task outputs
        self.fc_glaucoma = nn.Linear(self.num_channels*4, 2)   
        self.fc_diabetes = nn.Linear(self.num_channels*4, 2)   

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 177 x 128.

        Returns:
            out: (Variable) dimension batch_size x 1 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        #                                                  -> batch_size x 3 x 177 x 128
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 177 x 128
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels x 88 x 64
        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 88 x 64
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 44 x 32
        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 44 x 32
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*4 x 22 x 16

        # flatten the output for each image
        s = s.view(-1, int(self.input_width*self.input_height/64)*self.num_channels*4)           # batch_size x 22*16*self.num_channels*4

        # apply 2 fully connected layers with dropout
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
            p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4
        s_glaucoma = self.fc_glaucoma(s)                    # batch_size x 2
        s_diabetes = self.fc_diabetes(s)                    # batch_size x 2

        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        outputs = [F.log_softmax(s_glaucoma, dim=1), F.log_softmax(s_diabetes, dim=1)] # return F.sigmoid(s)

        return outputs


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    num_examples = outputs[0].size()[0]
    criterion = nn.CrossEntropyLoss()
    loss_glaucoma = criterion(outputs[0], labels[:, 0])/num_examples
    loss_diabetes = criterion(outputs[1], labels[:, 1])/num_examples
    loss = (loss_glaucoma+loss_diabetes)
    return [loss_glaucoma, loss_diabetes, loss]
    #loss = nn.CrossEntropyLoss()
    #return loss(outputs.view(num_examples, -1).squeeze(), labels)/num_examples
    #return -torch.sum(outputs[range(num_examples), labels])/num_examples

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 2 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy
    , 'precision': functions.getPrecision
    , 'recall': functions.getRecall
    , 'f1score': functions.getMacroFScore
    # could add more metrics such as accuracy for each token type
}
