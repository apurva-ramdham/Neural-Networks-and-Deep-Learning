import numpy as np
from random import shuffle
np.set_printoptions(threshold=np.inf)
def sigmoid(x):
    h = np.zeros_like(x)
    
    #############################################################################
    # TODO: Implement sigmoid function.                            #         
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    h = 1/(1+np.exp(-x))
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################
    return h 

def logistic_regression_loss_naive(W, X, y, reg):
    """
      Logistic regression loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c can be either 0 or 1.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Set the loss to a random number
    loss = None
    # Initialize the gradient to zero
    dW = None

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    C = W.shape[1]
    D = X.shape[1]
    N = X.shape[0]
    loss_class = np.zeros(C)
    dW = np.zeros_like(W)

    #One-hot encoding
    y_new = np.squeeze(np.eye(C)[y.reshape(-1)])

    #Computing loss function
    for k in range(C):
      for i in range(N):
        f_j = 0
        x_sample = X[i,:]
        for j in range(D):
          f_j += X[i,j] * W[j,k]                 
        h = sigmoid(f_j)
        loss_class[k] += y[i] * np.log(h) + (1-y[i]) * np.log(1-h)  
        dW[:,k] += (y_new[i,k]-h)*x_sample.T  
      loss_class[k] /= -N

    loss = np.sum(loss_class)
    dW /= N

    #Regularization
    loss += 0.5 * reg * np.sum(W*W)
    dW += reg*W
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW



def logistic_regression_loss_vectorized(W, X, y, reg):
    """
    Logistic regression loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Set the loss to a random number
    loss = None
    # Initialize the gradient to zero
    dW = None

    ############################################################################
    # TODO: Compute the logistic regression loss and its gradient using no    # 
    # explicit loops.                                       #
    # Store the loss in loss and the gradient in dW. If you are not careful   #
    # here, it is easy to run into numeric instability. Don't forget the     #
    # regularization!                                       #
    ############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    C = W.shape[1]
    D = X.shape[1]
    N = X.shape[0]
    loss = np.zeros(C)
    dW = np.zeros((D,C))

    #One-hot encoding
    y_new = np.squeeze(np.eye(C)[y.reshape(-1)])

    #Loss Computation
    f = X.dot(W)
    h = sigmoid(f)
    h = h.T
    loss = np.sum(np.log(h).dot(y) + np.log(1 - h).dot(1-y))
    loss /= -N

    #Grad Computation 
    dW = X.T.dot(y_new - sigmoid(f))
    dW /= N

    #Regularization
    loss += 0.5 * reg * np.sum(W*W)
    dW += reg*W
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW
