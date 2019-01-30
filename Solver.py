import numpy as np
import torch
import torch.nn as nn

class Solver(object):
    
    def __init__(self, net, data, **kwargs):

        self.net = net
        self.train_data = data['train_data']
        self.train_labels = data['train_labels']
        self.val_data = data['val_data']
        self.val_labels = data['val_labels']
        
        # Unpack arguments
        self.learning_rate = kwargs.pop("learning_rate",1e-3)
        self.num_epochs = kwargs.pop("num_epochs",2)
        self.batch_size = kwargs.pop("batch_size",64)
        
        # Define the loss function an optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)

        # Loss history and accuracy history
        self.loss_history = []
        self.acc_history = []
        
        
        
    def _step(self):
        # Number of training samples
        n_train = self.train_data.shape[0] 
        # Select the minibatch
        mask = np.random.choice(n_train, self.batch_size)
        x = torch.from_numpy(np.float32(self.train_data[mask])) # 64x15
        y = torch.from_numpy(np.argmax(self.train_labels[mask], axis=1)) # 64x1
        
        # Zero gradients
        self.optimizer.zero_grad()

        # Forward + Backward + optimize
        y_pred = self.net(x)
        loss = self.loss_fn(y_pred,y)        
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def check_accuracy(self,x, y, num_samples=None, batch_size = 64):
        """
        Returns the fraction of instances that were correctly classified by the model
        """
        N = x.shape[0] # number of samples of the input dataset
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples) # Subsample the data
            x = x[mask]
            y = y[mask]
            N = num_samples

        # Labels in a row
        labels = np.argmax(y,axis=1).ravel()

        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1

        y_pred = np.empty((num_batches,batch_size))
        y_pred[:] = np.nan    
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            input_data = torch.from_numpy(np.float32(x[start:end]))
            scores = self.net(input_data)
            _, predicted = torch.max(scores, 1)
            y_pred[i,:len(predicted)] = predicted.numpy()

        # Remove nan values
        y_pred = y_pred.ravel()
        y_pred = y_pred[np.logical_not(np.isnan(y_pred))]

        # Predict accuracy
        acc = np.mean(y_pred == labels)

        return acc

    
    def train(self):
        # Number of training samples
        n_train = self.train_data.shape[0] 
        # Number of iterations per epoch
        iters_per_epoch = int(n_train/self.batch_size) # 75
        
        # Initialize loss history and acc history
        loss_history = np.zeros((iters_per_epoch,self.num_epochs))
        acc_history = []
        
        # Loop over the dataset multiple times
        for epoch in range(self.num_epochs):

            for t in range(iters_per_epoch):
                loss = self._step()

                # Statistics
                loss_history[t,epoch]=loss.item()

                if(t%10 == 0): # Print each 9 minibatches
                    train_acc = self.check_accuracy(self.train_data, self.train_labels, num_samples = 1000)
                    acc_history.append(train_acc)

                    print('[%d, %5d] loss: %.3f train accuracy: %.2f' %
                          (epoch + 1, t + 1, loss_history[t,epoch] / 100, train_acc*100))

        self.loss_history = loss_history
        self.acc_history = acc_history
        print('Finished Training\n\n\n')
