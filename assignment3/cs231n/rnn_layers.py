from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################

    mh = np.dot(prev_h,Wh)              # internal state dependancy
    mx = np.dot(x,Wx)                   # input dependancy
    sx = mh + mx + b                    # add bias

    next_h = np.tanh(sx)                # directly use numpy tanh activation function
    #sxm = -2.0 * sx
    #sxe = np.exp(sxm)
    #sxn = sxe - 1.0
    #sxp = sxe + 1.0
    #sxpr = sxp**(-1.0)
    #mnh = sxn * sxpr
    #next_h = -1.0 * mnh
    #cache = prev_h,x,b,Wx,Wh,mh,mx,sx,sxm,sxe,sxn,sxp,sxpr,mnh
    cache = prev_h,x,b,Wx,Wh,mh,mx,sx


    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of in put data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    prev_h,x,b,Wx,Wh,mh, mx, sx  = cache
    #prev_h,x,b,Wx,Wh,mh, mx, sx, sxm, sxe, sxn, sxp, sxpr, mnh = cache

    #dmnh = -1.0 * dnext_h
    #dsxn = dmnh * sxpr
    #dsxpr = dmnh * sxn
    #dsxp = (-1.0 / (sxp**2)) * dsxpr
    #dsxe = dsxn + dsxp
    #dsxm = dsxe * np.exp(sxm)
    #dsx = -2.0 * dsxm
    dsx = (1 - (np.tanh(sx)**2)) * dnext_h  # use the analytic tanh derivative, 1-tahh^2(x)

    dx  = np.dot(dsx,Wx.T)
    dWx = np.dot(dsx.T,x).T
    dprev_h = np.dot(dsx,Wh.T)
    dWh =  np.dot(prev_h.T,dsx)
    db = np.sum(dsx,0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    N = x.shape[0]  # Batch size
    T = x.shape[1]  # number of timesteps
    H = h0.shape[1] # hidden state size

    h = np.zeros([N,T,H])       # placeholder for result
    cache = [None for j in range(T)]
    next_h = h0
    for i in range (T):
        next_h, cache_i = rnn_step_forward(x[:,i,:], next_h, Wx, Wh, b)
        h[:,i,:] = next_h
        cache[i] = cache_i
        #cache = prev_h, x, b, Wx, Wh, mh, mx, sx, sxm, sxe, sxn, sxp, sxpr, mnh
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    #dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################

    _, cx, _, _, _, _, _, _  = cache[0]
    #_, cx, _, _, _, _, _, _, _, _, _, _, _, _ = cache[0]
    D = cx.shape[1]  # input vector size
    N = dh.shape[0]  # Batch size
    T = dh.shape[1]  # number of timesteps
    H = dh.shape[2]  # hidden state size

    dx = np.zeros([N,T,D])
    dWx = np.zeros([D,H])
    dWh = np.zeros([H,H])
    db = np.zeros([H])

    ldh = np.copy(dh)               # make sure we don't mess with the original
    # loop over all timesteps, from last to first
    for i in range (T-1,-1,-1):
        dx_s, dh_s, dWx_s, dWh_s, db_s = rnn_step_backward(ldh[:,i,:], cache[i])
        dx[:,i,:] = dx_s            # update the input gradient
        dWx += dWx_s                # update the Wx gradient
        dWh += dWh_s                # update the Wh gradient
        db += db_s                  # update yje bias gradient
        if (i > 0):
            ldh[:, i-1, :] += dh_s  # update the hidden state gradient
        else:
            dh0 = dh_s              # log the initial hidden state gradient
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    V = W.shape[0]
    D = W.shape[1]

    #N = x.shape[0]
    #T = x.shape[1]
    #out = np.zeros([x.shape[0],x.shape[1],W.shape[1]])  #create the right shaped template
    #for i in range (N):
    #    for j in range (T):
    #        out[i,j,:] = W[x[i,j],:]        #plug in the vector matching this intege

    # simply use array indexing ...
    # out: [N,T,D]
    # x  : [N,T]
    #    so array indexing automatically iterates over all common indexes (N,T)
    out = W[x]

    cache = x,V,D
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    x,V,D = cache
    dW = np.zeros([V,D])

    #N = x.shape[0]
    #T = x.shape[1]
    #for i in range (N):
    #    for j in range (T):
    #        dW[x[i,j],:] += dout[i,j,:]
    np.add.at(dW, x, dout)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    #D = x.shape[1]
    H = prev_h.shape[1]
    # (1)
    mh = np.dot(prev_h, Wh)         # internal state dependancy
    mx = np.dot(x, Wx)              # input dependancy
    sx = mh + mx + b                # add bias

    sxi = sx[:,0:H]
    sxf = sx[:,H:2*H]
    sxo = sx[:,2*H:3*H]
    sxg = sx[:,3*H:4*H]

    # (2)
    #sxfa = np.apply_along_axis(sigmoid,1,sxf)
    sxfa = sigmoid(sxf)
    sxia = sigmoid(sxi)
    sxga = np.tanh(sxg)
    sxoa = sigmoid(sxo)

    # (3)
    pcf = prev_c * sxfa
    igw = sxia * sxga
    next_c = pcf + igw
    nct = np.tanh(next_c)
    next_h = nct * sxoa
    cache = next_c,next_h,prev_h,prev_c,x,b,Wx,Wh,mh,mx,sx,sxfa,sxia,sxga,sxoa,pcf,igw,nct
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    next_c,next_h,prev_h, prev_c, x, b, Wx, Wh, mh, mx, sx,\
                        sxfa, sxia, sxga, sxoa, pcf, igw, nct = cache
    H = prev_h.shape[1]
    sxg = sx[:, 3 * H:4 * H]

    dnct = dnext_h * sxoa
    # (3)
    dtnext_c = dnct * (1 - (np.tanh(next_c) ** 2))
    dinext_c = dnext_c + dtnext_c
    dpcf = dinext_c
    dprev_c = dpcf * sxfa
    # (2)
    dsxfa = dpcf * prev_c
    dsxf = dsxfa * sxfa * (1 - sxfa)

    digw = dinext_c
    dsxia = digw * sxga
    dsxga = digw * sxia
    dsxoa = dnext_h * nct

    dsxi = dsxia * sxia * (1-sxia)
    dsxg = dsxga * (1-(np.tanh(sxg)**2))
    dsxo = dsxoa * sxoa * (1-sxoa)
    dsx = np.concatenate((dsxi,dsxf,dsxo,dsxg),axis=1)
    # (1)
    dx = np.dot(dsx, Wx.T)
    dWx = np.dot(dsx.T, x).T
    dprev_h = np.dot(dsx, Wh.T)
    dWh = np.dot(prev_h.T, dsx)
    db = np.sum(dsx,0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    N = x.shape[0]  # Batch size
    T = x.shape[1]  # number of timesteps
    H = h0.shape[1]  # hidden state size

    h = np.zeros([N, T, H])  # placeholder for result
    cache = [None for j in range(T)]
    next_h = h0                     # initial Hidden state
    next_c = np.zeros(H)            # initial cell state
    for i in range(T):
        next_h,next_c, cache_i = lstm_step_forward(x[:, i, :], next_h,next_c, Wx, Wh, b)
        h[:, i, :] = next_h
        cache[i] = cache_i
        # cache = prev_h, x, b, Wx, Wh, mh, mx, sx, sxm, sxe, sxn, sxp, sxpr, mnh
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    _,_,_,_,x,_,_,_,_,_,_,_,_,_,_,_,_,_ = cache[0]
    D = x.shape[1]   # input vector size
    N = dh.shape[0]  # Batch size
    T = dh.shape[1]  # number of timesteps
    H = dh.shape[2]  # hidden state size

    dx = np.zeros([N,T,D])
    dWx = np.zeros([D,4*H])
    dWh = np.zeros([H,4*H])
    db = np.zeros([4*H])

    ldh = np.copy(dh)               # make sure we don't mess with the original
    dprev_c = np.zeros(H)
    # loop over all timesteps, from last to first
    for i in range (T-1,-1,-1):
        dx_s, dh_s, dprev_c, dWx_s, dWh_s, db_s = lstm_step_backward(ldh[:,i,:],dprev_c, cache[i])
        dx[:,i,:] = dx_s            # update the input gradient
        dWx += dWx_s                # update the Wx gradient
        dWh += dWh_s                # update the Wh gradient
        db += db_s                  # update yje bias gradient
        if (i > 0):
            ldh[:, i-1, :] += dh_s  # update the hidden state gradient
        else:
            dh0 = dh_s              # log the initial hidden state gradient
    ##############################################################################
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
