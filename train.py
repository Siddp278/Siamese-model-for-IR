import torch
import numpy as np


def inferece(y_pred, y):
        """ Performs inference """
        return (y_pred == y).sum().item()

def train_epoch(epochs, train_loader,
                model, optimizer, loss_fn, threshold):
    # Not implemented validation for each epoch
    loss_history = []
    for i in range(epochs):
        temp_loss = 0
        correct_total = 0
        for batch_x, batch_y, y_ in train_loader:
                # print(f"Iteration [{i+1}/{epochs}]  Training")
                q1, q2, y = batch_x.cuda(), batch_y.cuda(), y_.cuda()
                # print(q1.shape, q2.shape, y.shape)        
                # Reset the gardients 
                optimizer.zero_grad()

                # Model forward and predictions
                similarity = model(q1, q2)
                y_pred = (similarity > threshold).float() * 1
                # print(y_pred.shape, y.shape)
                correct = inferece(y_pred, y)
                correct_total += correct


                # Calculate the loss 
                loss = loss_fn(similarity, y)
                temp_loss += loss.item()

                # Calculate gradients by performign the backward pass
                loss.backward()
                        
                # Update weights
                optimizer.step()
        loss_history.append(temp_loss)
        print(f"Epoch: {i}, train_loss: {temp_loss}")
            
    # Not enabled learning rate scheduler  

    return loss_history