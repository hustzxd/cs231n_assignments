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
  dW = np.zeros(W.shape) # initialize the gradient as zero  (3073,10)
  #print(dW.shape)
  #print("..")

  # compute the loss and the gradient
  num_classes = W.shape[1]   #10
  num_train = X.shape[0]     #500
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

        # Compute gradients (one inner and one outer sum)
        # Wonderfully compact and hard to read
        dW[:, y[i]] -= X[i, :].T # this is really a sum over j != y_i
        dW[:, j] += X[i, :].T # sums each contribution of the x_i's

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Same with gradient
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)


  # Gradient regularization that carries through per https://piazza.com/class/i37qi08h43qfv?cid=118
  dW += reg*W
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

  #print(type(W))
  #print(type(X))

  num_classes = W.shape[1]
  num_train = X.shape[0]


  #print("shape")
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  correct_class_score = scores[np.arange(num_train), y]

  #print(scores.shape)
  #print(correct_class_score.shape)
  #print(type(scores))
  #print(type(correct_class_score))

  tmpMat = scores.T - correct_class_score + 1
  tmpMat = tmpMat.T
  tmpMat[np.arange(num_train), y] = 0

  margin = np.maximum(tmpMat, np.zeros((num_train, num_classes)))

  #print(margin.shape)
  loss = np.sum(margin)
  loss /= num_train
  loss += 0.5*reg*np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  # Binarize into integers
  binary = margin
  binary[margin > 0] = 1

  # Perform the two operations simultaneously
  # (1) for all j: dW[j,:] = sum_{i, j produces positive margin with i} X[:,i].T
  # (2) for all i: dW[y[i],:] = sum_{j != y_i, j produces positive margin with i} -X[:,i].T
  col_sum = np.sum(binary, axis=1)
  binary[np.arange(num_train), y] = -col_sum[range(num_train)]
  dW = np.dot(X.T, binary)

  # Divide
  dW /= num_train

  # Regularize
  dW += reg*W



  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
