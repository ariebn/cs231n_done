from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0, dropout_prob=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    ################# Dropouts
    W2s = W2.shape[0]
    nr = np.random.choice(W2s, (W2s*dropout_prob))   #elemetns to zero
    nmask = np.full(W2s,1)
    nmask[nr] = 0
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    # scores = np.maximum(X.dot(W1) + b1,0).dot(W2) + b2
    va = X.dot(W1)                                        #(1)
    vas = va + b1                                         #(2)
    vasr = np.maximum(vas,0)      # RelU                  #(3)
    #print (vasr.shape, nmask.shape)

    W2m = np.multiply(W2.T, nmask).T
    v2 = vasr.dot(W2m)                                    #(4)
    scores = v2 + b2                                      #(5)

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss

    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    cp = scores - np.max(scores)    #Range normalization  #(6)
    cpe = np.exp(cp)                                      #(7)
    cper = (cpe.T / np.sum(cpe, axis=1)).T                #(8)

    # Loss calculation
    # pick from each row the cell that corresponds to the correct class (Advanced indexing
    # cscores = vscores[np.arange(y.shape[0]), y]  # e.g. vscores[[0,1,2,...499],33]
    cscores = cper[np.arange(y.shape[0]), y]              #(9)
    lcscores = -1 * np.log(cscores)                       #(10)
    lossp =  np.sum(lcscores)                             #(11)
    lossa = lossp / N                                     #(12)

    # Add a regularization term
    rega = (np.sum(W2 * W2) + np.sum(W1 * W1))            #(13)
    regv = reg * rega                                     #(14)
    loss = lossa + regv                                   #(15)

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass

    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    # Gradient calculation
    # subract 1 from the correct class score, for gradient calculation (makes sum=0)
    icper = cper
    icper[np.arange(y.shape[0]), y] -= 1
    dW2 = np.dot(vasr.T, icper)
    db2 = np.sum(icper, axis=0)

    dvasr = np.dot(icper,W2.T)
    ru_mask = np.clip(np.sign(vas),0,1)
    dvas = dvasr * ru_mask                #1 if positive, 0 else

    dW1 = np.dot(X.T,dvas)
    db1 = np.sum(dvas,axis=0)

    # Average gradient across all training inputs

    dW2 /= N
    db2 /= N
    dW1 /= N
    db1 /= N

    # Regularization term for gradient
    dW1 += 2 * reg * W1
    dW2 += 2 * reg * W2

  # Dropouts
    dW1m = np.multiply(dW1, nmask)
    db1m = np.multiply(db1, nmask)
    dW2m = np.multiply(dW2.T, nmask).T

  # print (dW2.shape, db2.shape, dW1.shape, db1.shape)

   # print (dW1m.shape, dW2m.shape, dW2[100,5],np.sum(dW1), np.sum(dW2))

    #stop = r + g
    # return gradients via directory
    grads['W1'] = dW1m
    grads['W2'] = dW2m
    grads['b1'] = db1m
    grads['b2'] = db2

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False, dropout_prob=0.0):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    - dropout_prob: probability of dropout during training
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    learning_rate_initial = learning_rate

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      Xi = np.random.choice(X.shape[0],batch_size)
      X_batch = X[Xi]
      y_batch = y[Xi]
      #try data augmentaion,  mirroring the training images for some
      if (it % 3 == 99999):
        Xbr = np.reshape(X_batch, (batch_size, 32, 32,3))
        #numpy1.13     Xbr1 = np.flip(Xbr,2)             #flip horizontally
        indexer = [slice(None)] * Xbr.ndim
        indexer[2] = slice(None, None, -1)
        Xbr1 = Xbr[tuple(indexer)]

        X_batch = np.reshape(Xbr1, (batch_size, 3072))
      # reset learning rate to  initial value for 2nd half of epochs
      #if (it == num_iters/2 ):
      #  learning_rate = learning_rate_initial

      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg, dropout_prob=dropout_prob)
      loss_history.append(loss)           #append to list

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      #self.W = self.W - (grad * learning_rate)
      self.params['W1'] = self.params['W1'] - ( grads['W1'] * learning_rate )
      self.params['W2'] = self.params['W2'] - ( grads['W2'] * learning_rate )
      self.params['b1'] = self.params['b1'] - ( grads['b1'] * learning_rate )
      self.params['b2'] = self.params['b2'] - ( grads['b2'] * learning_rate )


      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 500 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate , after each epoch
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X,dropout_prob=0.0):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    H1 = np.maximum(X.dot(W1) + b1,0) * (1 - dropout_prob)
    scores = H1.dot(W2)+b2
    #scores = np.maximum(X.dot(W1) + b1,0).dot(W2) + b2
    y_pred = np.argmax(scores,axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


