import math
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


class SoftmaxClassifier:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.W = None
        self.initialize()

    def initialize(self):
        # TODO your code here
        # initialize the weight matrix (remember the bias trick) with small random variables
        # you might find torch.randn userful here *0.001
        # don't forget to set call requires_grad_() on the weight matrix,
        # as we will be taking its gradients during the learning process
        self.W = torch.randn(self.input_shape, self.num_classes) * 0.01
        """
        In PyTorch, an in-place operation refers to an operation that modifies the content 
        of a tensor without creating a new tensor. Instead of creating a new tensor to store
        the result, the operation directly modifies the existing tensor. This can be more 
        memory-efficient but can also have implications for gradient computation when
        requires_grad is set to True.

        When you perform in-place operations on a tensor with requires_grad=True, PyTorch 
        may not be able to accurately track the gradients through the in-place operation. 
        This is because in-place operations can change the tensor's data and computational 
        graph, and gradient tracking may not be reliable.
        """
        self.W.requires_grad_()

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        # TODO your code here
        # 0. compute the dot product between the input X and the weight matrix
        # you can use @ for this operation
        scores = X @ self.W
        # remember about the bias trick!
        # 1. apply the softmax function on the scores, see torch.nn.functional.softmax
        # think about on what dimension (dim parameter) you should apply this operation
        scores = torch.nn.functional.softmax(scores, dim=1)
        # * This will compute softmax along the rows, and the resulting tensor will have each row summing to 1.
        # 2. returned the normalized scores
        return scores

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        # TODO your code here
        # 0. compute the dot product between the input X and the weight matrix
        # 1. compute the prediction by taking the argmax of the class scores
        # you might find torch.argmax useful here.
        # think about on what dimension (dim parameter) you should apply this operation
        # ? the coef a means
        scores = torch.mm(X, self.W)  # or do X @ self.W
        label = torch.argmax(scores, dim=1)

        return label

    def cross_entropy_loss(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # TODO your code here
        # ! choose only the elements for the ground truth label
        y_pred_true = -y_pred[torch.arange(y_pred.shape[0]), y]
        # !!!!! LOG ALREADY APPLIED
        loss = torch.mean(y_pred_true)
        return loss

    def l2_regularization(self):
        copy = self.W.clone().detach()
        #? not copy in sum
        return torch.sum(self.W)

    def log_softmax(self, x: torch.Tensor) -> torch.Tensor:
        c = torch.reshape(torch.max(x, dim=1).values, (x.shape[0], 1))
        sum = torch.reshape(torch.sum(torch.exp(x-c), dim=1), (x.shape[0], 1))  # ! reshape for broadcasting
        return x - c - torch.log(sum)

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor, mean_image,
            **kwargs) -> dict:

        history = []
        bs = kwargs['bs'] if 'bs' in kwargs else 128
        reg_strength = kwargs['reg_strength'] if 'reg_strength' in kwargs else 1e3
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 100
        lr = kwargs['lr'] if 'lr' in kwargs else 1e-3
        print('hyperparameters: bs {:.1f} lr {:.4f}, reg {:.4f}, epochs {:.2f}'.format(bs, lr, reg_strength, epochs))

        for epoch in range(epochs):
            if epoch > 3:
                lr = lr * 0.1
                reg_strength *= 0.1
            # if epoch > 5:
            #     lr = lr * 0.1
            for ii in range((X_train.shape[0] - 1) // bs + 1):  # in batches of size bs
                # TODO your code here
                start_idx = ii * bs  # we are ii batches in, each of size bs
                end_idx = start_idx + bs   # get bs examples

                # get the training examples xb, and their corresponding annotations
                xb = X_train[start_idx:end_idx]
                yb = y_train[start_idx:end_idx]  # ! a 1d array with labels

                # plt.imshow(xb[0, :-1].reshape((32, 32, 3)))
                # plt.show()
                # apply the linear layer on the training examples from the current batch
                pred = torch.mm(xb, self.W)
                pred = self.log_softmax(pred)

                # compute the loss function
                # also add the L2 regularization loss (the sum of the squared weights)
                loss = self.cross_entropy_loss(pred, yb) + reg_strength * self.l2_regularization()
                history.append(loss.detach().numpy())

                # start backpropagation: calculate the gradients with a backwards pass
                loss.backward()

                # update the parameters
                with torch.no_grad():  # we don't want to track gradients
                    # take a step in the negative direction of the gradient, the learning rate defines the step size
                    self.W -= self.W.grad * lr

                    # ATTENTION: you need to explicitly set the gradients to 0 (let pytorch know that you are
                    # done with them).
                    self.W.grad.zero_()

        return history

    def get_weights(self, img_shape) -> np.ndarray:
        # TODO your code here
        W = self.W.detach().numpy()
        # 0. ignore the bias term
        W = W[:-1, :]
        # 1. reshape the weights to (*image_shape, num_classes)
        d1, d2, d3 = img_shape
        W = W.T.reshape(self.num_classes, d1, d2, d3)
        # you might find the transpose function useful here
        return W

    def load(self, path: str) -> bool:
        # TODO your code here
        # load the input shape, the number of classes and the weight matrix from a file
        # you might find torch.load useful here
        data = torch.load(path)
        self.W = data["weights"]
        # don't forget to set the input_shape and num_classes fields
        self.num_classes = data["no_classes"]
        self.input_shape = data["input_shape"]
        return True

    def save(self, path: str) -> bool:
        # TODO your code here
        # save the input shape, the number of classes and the weight matrix to a file
        # you might find torch useful for this
        torch.save({
            "input_shape": self.input_shape,
            "no_classes": self.num_classes,
            "weights": self.W
        }, path)
        # TODO your code here
        return True


#%%
