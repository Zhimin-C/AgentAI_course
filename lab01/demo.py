import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import cv2
import os

save_dir = './FashionMNIST/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

## Loading FashionMNIST dataset:
training_data = torchvision.datasets.FashionMNIST('./FashionMNIST/',train=True,download=True,transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.FashionMNIST('./FashionMNIST/',train=False,download=True,transform=torchvision.transforms.ToTensor())

## Preparing Data Loader:

trainDataLoader = torch.utils.data.DataLoader(training_data,batch_size=32,shuffle=True)
testDataLoader = torch.utils.data.DataLoader(test_data,batch_size=32)

classes = {0: 'T-shirt/top'         ###  Mapping labels to categorical class names 
,1: 'Trouser'
,2: 'Pullover'
,3: 'Dress'
,4: 'Coat'
,5: 'Sandal'
,6: 'Shirt'
,7: 'Sneaker'
,8: 'Bag'
,9: 'Ankle boot'}

## Displaying some sample images
images, labels = next(iter(trainDataLoader))
for i in range(3):
    plt.imshow(images[i].squeeze(), cmap = 'gray')
    plt.title(classes[int(labels[i])])
    plt.show()
    plt.close()


### Model Architecture:
class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear_relu_stack = torch.nn.Sequential(
                torch.nn.Linear(784, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 10),
            )

    def forward(self, layer):
        layer = layer.view(-1,784)
        layer = self.linear_relu_stack(layer)
        return layer

model = NN()

## Summary of Neural Network:
print("###### Model Summary ######")
print(model)


## Defining Optimizer and Loss 
## Optimizer:
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
## Using categorical cross entropy entropy as Loss:
Loss = torch.nn.CrossEntropyLoss()

### Training neural network:
train_loss_epochs = []
test_loss_epochs = []

for epoch in range(30):
    train_loss = 0.
    test_loss = 0.
    for train_data in trainDataLoader:
        optimizer.zero_grad() 
        train_images, labels = train_data
        y_pred = model(train_images)
        l = Loss(y_pred,labels)         ## computing loss
        l.backward()
        train_loss = train_loss + l.item()
        optimizer.step()
    for test_data in testDataLoader:
        with torch.no_grad():
            test_images, labels = test_data
            y_pred = model(test_images)
            l = Loss(y_pred,labels)
            test_loss = test_loss + l.item()
    train_loss = train_loss/len(trainDataLoader)
    test_loss = test_loss/len(testDataLoader)
    train_loss_epochs.append(train_loss)
    test_loss_epochs.append(test_loss)
    print("Training loss: {}, Test loss: {} on epoch: {}".format(train_loss, test_loss, epoch))

### Plotting the training and test loss
plt.grid(True)
plt.plot(np.arange(30), train_loss_epochs, label='Training Loss')
plt.plot(np.arange(30), test_loss_epochs, label='Testing Loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig(save_dir + 'loss_plot.png')
plt.close()

### Model's Test Accuracy:
correct_pred = 0.
total_images = 0.
for test_images,labels in testDataLoader:
    for i in range(len(labels)):
        with torch.no_grad():
            pred = model(test_images[i].view(1, 784))
            _, pred_class = torch.max(pred.data, 1)
            true_label = labels.data[i]
            if true_label == pred_class:
                correct_pred += 1
            total_images += 1

print("Total Number of testing images: {}".format(total_images))
print("Testing Accuracy of the model: {}".format(correct_pred/total_images))

### Printing the probabilities of test images:
test_images_iter, iter_labels = next(iter(testDataLoader))
softmax_probs = []
probs = []
for i in range(3):
    with torch.no_grad():
        prob = model(test_images_iter[i])
        probs.append(prob)
        softmax_probs.append(torch.nn.Softmax(dim=1)(prob))
        # print("Probability Tensor after softmax for {}: {}".format(classes[int(iter_labels[i])], softmax_probs[i]))
    plt.title(classes[int(iter_labels[i])])
    plt.imshow(test_images_iter[i].squeeze().numpy(),cmap='gray')
    plt.show()  
    plt.savefig(save_dir + 'groundtruth_test_image_{}.png'.format(i))
    plt.close()

### Probabilities Visualization before softmax:
fig = plt.figure(figsize=(30,10))

for i in range(1, 4):
    plt.subplot(1, 3, i)
    plt.grid(True)
    plt.bar(classes.values(), probs[i-1].squeeze(), color ='blue')
    plt.xlabel("Categorical Classes")
    plt.ylabel("Probability")
    plt.title(classes[int(iter_labels[i-1])])

plt.show()
plt.savefig(save_dir + 'probability_plot.png')
plt.close()

## Probabilies Visualization after softmax:
fig = plt.figure(figsize=(30,10))
for i in range(1, 4):
    plt.subplot(1, 3, i)
    plt.grid(True)
    plt.bar(classes.values(), softmax_probs[i-1].squeeze(), color ='blue')
    plt.xlabel("Categorical Classes")
    plt.ylabel("Probability")
    plt.title(classes[int(iter_labels[i-1])])
plt.savefig(save_dir + 'softmax_probability_plot.png')
plt.close()