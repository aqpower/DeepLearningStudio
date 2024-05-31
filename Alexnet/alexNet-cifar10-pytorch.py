#!/usr/bin/env python
# coding: utf-8

# In[46]:


import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms
from torchvision.datasets import CIFAR10


# In[47]:


# Mini-Batch Gradient Descent
batch_size = 128

learning_rate = 0.0001
num_epochs = 25
num_classes = 10

# Percentage of training set to use as validation
n_valid = 0.2

classes = (
    "Airplane",
    "Car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# In[48]:


# Define the transformation to apply to the data
transform = transforms.Compose(
    [
        transforms.Resize(224),  # fit the image to AlexNet input size
        transforms.ToTensor(),  # Convert PIL Image to tensor  [0.0, 1.0]
        # Normalize the data with the mean and standard deviation of the dataset
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(1.0, 1.0, 1.0)),
    ]
)

train_data = CIFAR10(root="./data", train=True, download=True, transform=transform)
test_data = CIFAR10(root="./data", train=False, transform=transform)

# Get indices for training_set and validation_set
n_train = len(train_data)
indices = list(range(n_train))
np.random.shuffle(indices)
split = int(np.floor(n_valid * n_train))
train_idx, valid_idx = indices[split:], indices[:split]

# Define samplers for obtaining training and validation
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# load the CIFAR-10 dataset
train_loader = DataLoader(
    train_data, batch_size=batch_size, sampler=train_sampler, num_workers=2
)

valid_loader = DataLoader(
    train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=2
)

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)


# In[49]:


# Checking the dataset size
def check_dataset(loader, set_name):
    print(f"{set_name} Set:")
    images, labels = next(iter(loader))
    print("image size per batch", images.size())
    print("label size per batch", labels.size())


check_dataset(train_loader, "Training")
check_dataset(test_loader, "Testing")


# ![alexnet-paper.png](https://jgox-image-1316409677.cos.ap-guangzhou.myqcloud.com/blog/alexnet-paper.png)
# 

# In[50]:


class AlexNet(nn.Module):

    def __init__(self, num_classes=num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # groups=2 means that the input is split into 2 groups and the convolution is applied to each group
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# In[51]:


model = AlexNet()
model.to(device)

# from torchsummary import summary

# summary(model, (3, 224, 224))


loss_fn = loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[52]:


def eval_model(model, data_loader):
    model.eval()
    class_total = [0.0 for _ in range(num_classes)]
    class_correct = [0.0 for _ in range(num_classes)]
    sum_loss, num_correct, num_examples = 0.0, 0.0, 0

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)

            # compute the model output
            outputs = model(features)
            _, predicted_labels = torch.max(outputs, 1)

            loss = F.cross_entropy(outputs, targets, reduction="sum")
            sum_loss += loss.item()

            # stat the correct radix
            num_examples += targets.size(0)
            num_correct += (predicted_labels == targets).sum().item()

            # compute each class 's correct count
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_correct[label] += (predicted_labels[i] == label).item()
                class_total[label] += 1

    accuracy = num_correct / num_examples * 100
    avg_loss = sum_loss / num_examples

    return {
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "class_correct": class_correct,
        "class_total": class_total,
    }


# In[53]:


# train model
def train():
    log_dict = {
        "train_loss_per_batch": [],
        "train_acc_per_epoch": [],
        "valid_acc_per_epoch": [],
        "train_loss_per_epoch": [],
        "valid_loss_per_epoch": [],
        "valid_loss_min": np.Inf,
    }
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader, 0):
            features = features.to(device)
            targets = targets.to(device)

            # FORWARD AND BACK PROP
            outputs = model(features)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # logging
            log_dict["train_loss_per_batch"].append(loss.item())
            if not batch_idx % 50:
                print(
                    f"Epoch: {epoch+1:03d}/{num_epochs:03d} | Batch {batch_idx:04d}/{len(train_loader):04d} | Loss: {loss:.4f}"
                )

        model.eval()

        with torch.set_grad_enabled(False):
            train_eval_res = eval_model(model, train_loader)
            train_acc = train_eval_res["accuracy"]
            train_loss = train_eval_res["avg_loss"]
            print(
                f"#Epoch: {epoch+1:03d}/{num_epochs:03d} | Train. Acc.: {train_acc:.3f}% | Loss: {train_loss:.3f}"
            )
            log_dict["train_loss_per_epoch"].append(train_loss)
            log_dict["train_acc_per_epoch"].append(train_acc)

            valid_eval_res = eval_model(model, valid_loader)
            valid_acc = valid_eval_res["accuracy"]
            valid_loss = valid_eval_res["avg_loss"]
            log_dict["valid_loss_per_epoch"].append(valid_loss)
            log_dict["valid_acc_per_epoch"].append(valid_acc)
            print(
                f"#Epoch: {epoch+1:03d}/{num_epochs:03d} | Valid. Acc.: {valid_acc:.3f}% | Loss: {valid_loss:.3f}"
            )
            if valid_loss <= log_dict["valid_loss_min"]:
                print(
                    f"#Validation loss decreased ({log_dict['valid_loss_min']:.6f} --> {valid_loss:.6f}). Saving model ..."
                )
                torch.save(model.state_dict(), "model_cifar.pt")
                log_dict["valid_loss_min"] = valid_loss

        print(f"Time elapsed: {(time.time() - start_time) / 60:.2f} min")

    print(f"Total Training Time: {(time.time() - start_time)/ 60:.2f} min")
    return log_dict


log_dict = train()


# In[54]:


model.load_state_dict(torch.load("model_cifar.pt"))


# In[55]:


def plot_training_metrics(log_dict: dict):
    loss_list = log_dict["train_loss_per_batch"]
    train_acc = log_dict["train_acc_per_epoch"]
    valid_acc = log_dict["valid_acc_per_epoch"]
    running_avg_loss = np.convolve(loss_list, np.ones(200) / 200, mode="valid")

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # plot training loss
    axs[0].plot(loss_list, label="Minibatch Loss")
    axs[0].plot(running_avg_loss, label="Running Average Loss", linewidth=2)
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Cross Entropy Loss")
    axs[0].set_title("Training Loss", fontsize=14, pad=15)
    axs[0].legend(loc="best")

    # plot training accuracy
    axs[1].plot(
        np.arange(1, num_epochs + 1),
        train_acc,
        label="Training Accuracy",
        color="blue",
        markersize=6,
        linewidth=2,
    )
    axs[1].plot(
        np.arange(1, num_epochs + 1),
        valid_acc,
        label="Valid Accuracy",
        color="red",
        markersize=6,
        linewidth=2,
    )
    axs[1].set_xticks(np.arange(1, num_epochs + 1, 2))
    axs[1].xlim = (0, num_epochs + 1)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].set_title("Accuracy", fontsize=14, pad=15)
    axs[1].legend(loc="best")
    # axs[1].grid(True)

    fig.savefig("training_performance.svg", format="svg")
    fig.show()


plot_training_metrics(log_dict)


# In[ ]:





# In[58]:


with torch.set_grad_enabled(False):
    test_eval_res = eval_model(model, test_loader)
    test_loss = test_eval_res["avg_loss"]
    test_acc = test_eval_res["accuracy"]
    class_correct = test_eval_res["class_correct"]
    class_total = test_eval_res["class_total"]

print(f"Test Loss: {test_loss:.4f}\n")
for i in range(num_classes):
    print(
        "Test Accuracy of %8s: %2d%% (%2d/%2d)"
        % (
            classes[i],
            100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]),
            np.sum(class_total[i]),
        )
    )
print(f"\nTest Accuracy (Overall): {test_acc:.2f}%")


# In[57]:


def plot_random_images_with_predictions(model, data_loader, classes):
    # step1: get 10 sample images from the data loader
    images, labels = next(iter(data_loader))
    images, labels = images[:10], labels[:10]

    images = images.to(device)
    labels = labels.to(device)

    # step2: get model predictions and calculate accuracy
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    correct_count = (predicted == labels).sum().item()
    accuracy = correct_count / len(labels) * 100

    # step3: plot the images with the predicted labels
    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(
        f"10 CIFAR-10 Images on Test Dataset\nAccuracy: {accuracy:.2f}%",
        fontsize=16,
        fontweight=600,
    )

    for i in range(10):
        ax = axes[i // 5, i % 5]
        img = np.transpose(images[i].numpy(), (1, 2, 0))

        ax.imshow(img)
        ax.axis("off")

        color = "blue" if predicted[i] == labels[i] else "red"
        ax.set_title(
            f"True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}",
            fontsize=12,
            color=color,
            y=-0.25,
        )

    plt.savefig("cifar10_predictions.svg", format="svg")
    plt.show()


plot_random_images_with_predictions(model, test_loader, classes)

