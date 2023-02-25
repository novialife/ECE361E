import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import argparse
from thop import profile
from torchsummary import summary


# Argument parser
parser = argparse.ArgumentParser(description='ECE361E HW3 - Starter PyTorch code')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train')
args = parser.parse_args()

# Always make assignments to local variables from your args at the beginning of your code for better
# control and adaptability
num_epochs = args.epochs
batch_size = args.batch_size

# Each experiment you will do will have slightly different results due to the randomness
# of the initialization value for the weights of the model. In order to have reproducible results,
# we have fixed a random seed to a specific value such that we "control" the randomness.
random_seed = 1
torch.manual_seed(random_seed)

# CIFAR10 Dataset (Images and Labels)
train_dataset = dsets.CIFAR10(root='data', train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]), download=True)

test_dataset = dsets.CIFAR10(root='data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]))

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# TODO: Get VGG11 model
from models import vgg11_pt
model = vgg11_pt.VGG11()

# TODO: Put the model on the GPU
device = torch.device("cuda:0")
model = model.to(device)

# Print model summary
print(summary(model, (3, 32, 32)))
flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).to(device),))
print('FLOPs: ', flops)
print('Params: ', params)

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters())

training_losses = []
testing_losses = []
training_accuracies = []
testing_accuracies = []
total_training_time = 0

starter = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Training loop
for epoch in range(num_epochs):
    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    
    # Sets the model in training mode.
    model = model.train()
    starter.record()

    for batch_idx, (images, labels) in enumerate(train_loader):
        # TODO: Put the images and labels on the GPU
        images = images.to(device)
        labels = labels.to(device)

        # Sets the gradients to zero
        optimizer.zero_grad()
        # The actual inference
        outputs = model(images)
        # Compute the loss between the predictions (outputs) and the ground-truth labels
        loss = criterion(outputs, labels)
        # Do backpropagation to update the parameters of your model
        loss.backward()
        # Performs a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.item()
        # The outputs are one-hot labels, we need to find the actual predicted
        # labels which have the highest output confidence
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        # Print every 100 steps the following information
        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                             len(train_dataset) // batch_size,
                                                                             train_loss / (batch_idx + 1),
                                                                             100. * train_correct / train_total))
    
    end.record()
    training_losses.append(train_loss / (batch_idx + 1))
    training_accuracies.append(100. * train_correct / train_total)
    torch.cuda.synchronize()
    total_training_time += starter.elapsed_time(end)
    
    # Testing phase
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # TODO: Put the images and labels on the GPU
            images = images.to(device)
            labels = labels.to(device)

            # Perform the actual inference
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    print('Test loss: %.4f Test accuracy: %.2f %%' % (test_loss / (batch_idx + 1),100. * test_correct / test_total))
    testing_losses.append(test_loss / (batch_idx + 1))
    testing_accuracies.append(100. * test_correct / test_total)
    
# TODO: Save the PyTorch model in .pt format
torch.save(model.state_dict(), "vgg11.pt")

print('Total training time: %.2f seconds' % (total_training_time/1000))
# Print the amount of model parameters
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
print(model)

with open('vgg11.txt', 'w') as f:
    f.write('Total training time: %.2f seconds\n' % (total_training_time/1000))
    f.write('Number of model parameters: {}\n'.format(sum([p.data.nelement() for p in model.parameters()])))
    f.write(str(model))
    f.write('\nTraining losses: {}\n'.format(training_losses))
    f.write('Training accuracies: {}\n'.format(training_accuracies))
    f.write('Testing losses: {}\n'.format(testing_losses))
    f.write('Testing accuracies: {}\n'.format(testing_accuracies))