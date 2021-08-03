"""
Speech Command Recognition with torchaudio
******************************************
This tutorial will show you how to correctly format an audio dataset and
then train/test an audio classifier network on the dataset.
Colab has GPU option available. In the menu tabs, select “Runtime” then
“Change runtime type”. In the pop-up that follows, you can choose GPU.
After the change, your runtime should automatically restart (which means
information from executed cells disappear).
First, let’s import the common torch packages such as
`torchaudio <https://github.com/pytorch/audio>`__ that can be installed
by following the instructions on the website.
"""

# Uncomment the line corresponding to your "runtime type" to run in Google Colab

# CPU:
# !pip install pydub torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# GPU:
# !pip install pydub torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

from datetime import datetime
import torch

import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

import matplotlib.pyplot as plt
import IPython.display as ipd
from datetime import datetime as dt

from tqdm import tqdm


from pyutilities import save_ckp
from audio_network import M5
import torchvision
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf

writer = SummaryWriter('log_dir')
######################################################################
# Let’s check if a CUDA GPU is available and select our device. Running
# the network on a GPU will greatly decrease the training/testing runtime.
#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


######################################################################
# Importing the Dataset
# ---------------------
#
# We use torchaudio to download and represent the dataset. Here we use
# `SpeechCommands <https://arxiv.org/abs/1804.03209>`__, which is a
# datasets of 35 commands spoken by different people. The dataset
# ``SPEECHCOMMANDS`` is a ``torch.utils.data.Dataset`` version of the
# dataset. In this dataset, all audio files are about 1 second long (and
# so about 16000 time frames long).
#
# The actual loading and formatting steps happen when a data point is
# being accessed, and torchaudio takes care of converting the audio files
# to tensors. If one wants to load an audio file directly instead,
# ``torchaudio.load()`` can be used. It returns a tuple containing the
# newly created tensor along with the sampling frequency of the audio file
# (16kHz for SpeechCommands).
#
# Going back to the dataset, here we create a subclass that splits it into
# standard training, validation, testing subsets.
#

from torchaudio.datasets import SPEECHCOMMANDS
import os


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


# Create training and testing split of the data. We do not use validation in this tutorial.
train_set = SubsetSC("training")
test_set = SubsetSC("testing")

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]


######################################################################
# A data point in the SPEECHCOMMANDS dataset is a tuple made of a waveform
# (the audio signal), the sample rate, the utterance (label), the ID of
# the speaker, the number of the utterance.
#

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.plot(waveform.t().numpy())
plt.title("Audio waveform of "+label)
plt.show()

######################################################################
# Let’s find the list of labels available in the dataset.
#

labels = sorted(list(set(datapoint[2] for datapoint in train_set)))

print("labels",labels)


######################################################################
# The 35 audio labels are commands that are said by users. The first few
# files are people saying “marvin”.
#

waveform_first, *_ = train_set[0]
ipd.Audio(waveform_first.numpy(), rate=sample_rate)

waveform_second, *_ = train_set[1]
ipd.Audio(waveform_second.numpy(), rate=sample_rate)


######################################################################
# The last file is someone saying “visual”.
#

waveform_last, *_ = train_set[-1]
ipd.Audio(waveform_last.numpy(), rate=sample_rate)


######################################################################
# Formatting the Data
# -------------------
#
# This is a good place to apply transformations to the data. For the
# waveform, we downsample the audio for faster processing without losing
# too much of the classification power.
#
# We don’t need to apply other transformations here. It is common for some
# datasets though to have to reduce the number of channels (say from
# stereo to mono) by either taking the mean along the channel dimension,
# or simply keeping only one of the channels. Since SpeechCommands uses a
# single channel for audio, this is not needed here.
#

new_sample_rate = 8000
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)

ipd.Audio(transformed.numpy(), rate=new_sample_rate)


######################################################################
# We are encoding each word using its index in the list of labels.
#


def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


word_start = "yes"
index = label_to_index(word_start)
word_recovered = index_to_label(index)

print(word_start, "-->", index, "-->", word_recovered)


######################################################################
# To turn a list of data point made of audio recordings and utterances
# into two batched tensors for the model, we implement a collate function
# which is used by the PyTorch DataLoader that allows us to iterate over a
# dataset by batches. Please see `the
# documentation <https://pytorch.org/docs/stable/data.html#working-with-collate-fn>`__
# for more information about working with a collate function.
#
# In the collate function, we also apply the resampling, and the text
# encoding.
#


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


batch_size = 256

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)





model = M5(n_input=transformed.shape[0], n_output=len(labels))
model.to(device)
# print(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


n = count_parameters(model)
print("Number of parameters: %s" % n)


######################################################################
# We will use the same optimization technique used in the paper, an Adam
# optimizer with weight decay set to 0.0001. At first, we will train with
# a learning rate of 0.01, but we will use a ``scheduler`` to decrease it
# to 0.001 during training after 20 epochs.
#

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10

#print preview the state dict and optimizer's state dict
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])




######################################################################
# Training and Testing the Network
# --------------------------------
#
# Now let’s define a training function that will feed our training data
# into the model and perform the backward pass and optimization steps. For
# training, the loss we will use is the negative log-likelihood. The
# network will then be tested after each epoch to see how the accuracy
# varies during the training.
#


def train(model, epoch, log_interval,checkpoint_dir):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        #plot model in tensorboard
        # writer.add_graph(model,data.reshape[-1])
        # print("Data Shape",data.shape)
        # writer.add_graph(model,data.reshape(-1,16000))
    
        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        #tensorboard preview
        #write loss function on tensorboard
        #either plot with epoch or step :batch_idx
        writer.add_scalar("Loss/train", loss, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

            #saving on checkpoint
            checkpoint_profile = {
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }
            save_ckp(checkpoint_profile,checkpoint_dir, epoch)
            


        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())

        
    # grid = torchvision.utils.make_grid(data)
    # writer.add_image('Images',grid,epoch)
    
        
    # writer.close()

######################################################################
# Now that we have a training function, we need to make one for testing
# the networks accuracy. We will set the model to ``eval()`` mode and then
# run inference on the test dataset. Calling ``eval()`` sets the training
# variable in all modules in the network to false. Certain layers like
# batch normalization and dropout layers behave differently during
# training so this step is crucial for getting correct results.
#


# def number_of_correct(pred, target):
#     # count number of correct predictions
#     return pred.squeeze().eq(target).sum().item()


# def get_likely_index(tensor):
#     # find most likely label index for each element in the batch
#     return tensor.argmax(dim=-1)


# def test(model, epoch):
#     model.eval()
#     correct = 0
#     for data, target in test_loader:

#         data = data.to(device)
#         target = target.to(device)

#         # apply transform and model on whole batch directly on device
#         data = transform(data)
#         output = model(data)

#         pred = get_likely_index(output)
#         correct += number_of_correct(pred, target)

#         # update progress bar
#         pbar.update(pbar_update)

#     print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")


######################################################################
# Finally, we can train and test the network. We will train the network
# for ten epochs then reduce the learn rate and train for ten more epochs.
# The network will be tested after each epoch to see how the accuracy
# varies during the training.
#

log_interval = 20
n_epoch = 50

pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []


checkpoint_dir = os.path.abspath("saved_checkpoints")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# The transform needs to live on the same device as the model and the data.
transform = transform.to(device)
with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval,checkpoint_dir)
        # test(model, epoch)
        scheduler.step()

# Let's plot the training loss versus the number of iteration.
# plt.plot(losses);
# plt.title("training loss");


######################################################################
# The network should be more than 65% accurate on the test set after 2
# epochs, and 85% after 21 epochs. Let’s look at the last words in the
# train set, and see how the model did on it.
#


