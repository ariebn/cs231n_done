import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #print ('dW shape= ', dW.shape)
  #print ('X[i] shape= ',X[2].shape)

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    # (10,) vector
    correct_class_score = scores[y[i]]
    missclass = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue     # goto next loop iteration...
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        missclass +=1     #count the number of missclasifications for this image
        dW[:, j] += X[i]   #update the gradient of incorrect-class-weight
      #
    dW[:, y[i]] -= missclass * X[i]   #update gradient of correct-class weight

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  # do the same averaging to the gradient
  dW /= num_train
  dW += 2 * reg * W                   # Regularization term for gradient

  # Add regularization to the loss
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #num_classes = W.shape[1]
  num_train = X.shape[0]
  vscores = X.dot(W)        # raw score matrix: for each of 500 tests, 10 scores

  #pick from each row the cell that corresponds to the correct class (Advanced indexing
  cscores = vscores[np.arange(y.shape[0]),y]    # e.g. vscores[[0,1,2,...499],33]

  # subtract from full score matrix the 'correct class' value, add 1, clip to 0, sum
  margins = np.clip((((vscores.T-cscores.T).T) + 1),0,None)
  loss = np.sum(margins)

  # Average per training set count
  loss /= num_train
  # compensate (Zero) the correct class score contribution, since we wrongly summed 1 for it
  loss -=1


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  # Gradient
  # use the margins array derived as part of the loss calculation

  # bdls: for each test -> is each class a mis-class error contributor (0/1)?
  bdls = (margins > 0).astype(int)

  # nz: (500,) : for each test, the number of misses contribs
  nz = np.sum(bdls, axis=1)
  nz -= 1  # decrement by 1 to null the correct class

  # Now plant the nz entry into the correct_class entry in bdls, as negative sign
  bdls[np.arange(y.shape[0]), y] = -1 * (nz)
  #print('bdls', bdls.shape)

  # ******************************* this is the key trick ************************
  # dot multiply the test data (500,3073) by the 'coerrection per test' matrix (500,10)
  dW = X.T.dot(bdls)
  # ******************************************************************************

  ##zz = np.zeros([X.shape[0], X.shape[1], W.shape[1]])
  ##mX = (X.T + zz.T).T  # just a duplicate of the image dataset, 10 times
  ##vbd = bdls[:,np.newaxis,:]
  ##zX = mX * vbd
  # sum all rows (cummulative gradient change
  ##szX = np.sum(zX,axis=0)

  # do the same averaging to the gradient
  dW /= num_train


  # add a regularization term to loss and gradient
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
