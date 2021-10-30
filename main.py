import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import ToTensor, Compose

import random
import string

import numpy as np

import math

all_letters = list(string.ascii_letters+'?!",.:()`’[];-—“”‘/£"èé&íáó¡úÁ¿ñ«» '+'\n'+'\t'+"'1234567890")

def number_to_tensor(number):
    Return = np.zeros(len(all_letters), dtype=np.float32)
    Return[number-1] = 1
    return Return

def letter_to_tensor(letter):
    return number_to_tensor(all_letters.index(letter))

def word_to_tensor(word):
    return np.array([letter_to_tensor(word[i]) for i in range(len(word))])

class Dataset(torch.utils.data.Dataset):
    def __init__(self, max_sample_size, transforms):
        super().__init__()

        self.dataset = []
        self.transforms = transforms

        file = open("dataset.txt", "r")

        encrypted_data = ""
        key = random.randint(0, int(len(all_letters)/2))
        other_letters = all_letters[key:] + all_letters[:key]

        check = 0
        data_size = max_sample_size
        for line in file:
            check += len(line)
            encrypted_data = ""
            for i in range(len(line)):
                if i % data_size == 0 and i != 0:
                    self.dataset.append((word_to_tensor(encrypted_data).reshape(max_sample_size, len(all_letters)), number_to_tensor(key)))
                    key = random.randint(0, len(all_letters))
                    other_letters = all_letters[key:] + all_letters[:key]
                    encrypted_data = ""

                encrypted_data += other_letters[all_letters.index(line[i])]

        file.close()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.transforms:
            return (self.transforms(self.dataset[idx][0]), self.dataset[idx][1])
        return self.dataset[idx]

step_size = 100
dataset = Dataset(step_size, transforms = ToTensor())

batch_size = 10
dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = nn.Sequential(
    nn.Conv1d(step_size, 75, kernel_size=1),
    nn.LeakyReLU(),

    nn.Conv1d(75, 50, kernel_size=1),
    nn.LeakyReLU(),

    nn.Conv1d(50, 25, kernel_size=1),
    nn.LeakyReLU(),

    nn.Flatten(),

    nn.Linear(2500, 100),
    nn.LeakyReLU(),

    nn.Sigmoid()
)

def eval(input, key):
    other_letters = all_letters[key:] + all_letters[:key]

    data = []
    for In in input:
        encrypted_data = ''.join(other_letters[all_letters.index(In[i])] for i in range(len(In)))
        data.append(word_to_tensor(encrypted_data).reshape(step_size, len(all_letters)))

    data = torch.tensor(np.asarray(data, dtype=np.float32))
    output = model(data)
    preds, accs = [], []
    for out in output:
        _, pred = torch.topk(out, 1)
        preds.append((len(input)*len(all_letters))-math.sqrt(((key-(pred.item()+1)))**2))
        accs.append(1 if (pred.item()+1) == key else 0)
    return preds, accs

def fit(num_epochs):
    losses, acc = [], []

    opt = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.BCELoss()

    for epoch in range(num_epochs):
        epoch_loss, epoch_preds, epoch_acc = [], [], []
        for i, (data, label) in enumerate(dataset):
            output = model(data.reshape(-1, step_size, len(all_letters))).reshape(-1, len(all_letters))
            loss = loss_fn(output, label)

            loss.backward()
            opt.step()
            opt.zero_grad()

            epoch_loss.append(loss.item())

        for key in range(len(all_letters)):
            Input = ["World War II or the Second World War, often abbreviated as WWII or WW2, was a global war that lasted", "I have a model that works perfectly when there are multiple input. However, if there is only one day", "Though not the first fictional detective, Sherlock Holmes is arguably the best known. By the 1990s h"]
            preds, accs = eval(Input, key)
            epoch_preds.append(preds)
            epoch_acc.append(accs)

        print("Epoch [{}/{}] with an average Loss of {} and an Outward Accuracy of {} and an Accuracy of {}".format(
        epoch+1, num_epochs, sum(epoch_loss) / len(epoch_loss), sum(np.asarray(epoch_preds).reshape(-1)) / len(np.asarray(epoch_preds).reshape(-1)), sum(np.asarray(epoch_acc).reshape(-1)) / len(np.asarray(epoch_acc).reshape(-1))
        ))

        losses.append(sum(epoch_loss) / len(epoch_loss))
        acc.append(sum(np.asarray(epoch_acc).reshape(-1)) / len(np.asarray(epoch_acc).reshape(-1)))

    return losses, acc

losses, acc = fit(2)
