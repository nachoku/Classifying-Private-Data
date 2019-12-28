from models import CNN
import os
import torch
from torch import optim, nn
from torchvision import transforms, datasets
import torchvision
import common



#DIRECTORY SETTINGS
os.chdir("..")#Go up two directories
DATA_DIR = 'data/bullying'
SAVE_DIR = 'models'
PRE_TRAINED_MODEL_SAVE_PATH=os.path.join(SAVE_DIR, 'base.pt')
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'pre_trained.pt')
# print(os.path.abspath(os.path.join(DIR, os.pardir)))

#Pointing to Data Directory

os.path.abspath(os.curdir)

#HYPERPARAMETERS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Normal
EPOCHS=100
BATCH_SIZE = 32
criterion = nn.CrossEntropyLoss()
ADAM_OPTIMISER=True
LEARNING_RATE=0.001
MOMENTUM=0.5


#1. DATA INPUT

#Load Pre-Trained model
model=CNN().to(device)
model.load_state_dict(torch.load(PRE_TRAINED_MODEL_SAVE_PATH))

#Load training data
train_transforms = transforms.Compose([
                           transforms.Resize(256),
                           transforms.RandomHorizontalFlip(30),
                           transforms.RandomRotation(10),
                           transforms.RandomCrop(256),
                           transforms.ToTensor(),
                           transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
                       ])

test_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
                       ])



#Directory import
train_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), train_transforms)
train_data, valid_data = torch.utils.data.random_split(train_data, [int(len(train_data)*0.9), len(train_data) - int(len(train_data)*0.9)])
test_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), test_transforms)

#Existing Dataset Import
train_data = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=train_transforms)
train_data, valid_data = torch.utils.data.random_split(train_data, [int(len(train_data)*0.9), len(train_data) - int(len(train_data)*0.9)])
test_data = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=test_transforms)


print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')


train_iterator = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
valid_iterator = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE)
test_iterator = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)



#2. ANALYSIS
if(ADAM_OPTIMISER):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
else:
    optimizer = optim.SGD(model.classifier.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

#Redefine model parameters to match current training set
for param in model.features.parameters():#Remove gradients
    param.requires_grad = False
model.classifier[6].out_features=9

#Train
best_valid_loss = float('inf')
for epoch in range(EPOCHS):
    print(epoch)
    train_loss, train_acc = common.train(model, device, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = common.evaluate(model, device, valid_iterator, criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(
        f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:05.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:05.2f}% |')




#3. OUTPUT

model.load_state_dict(torch.load(MODEL_SAVE_PATH))#Load best weights from file
test_loss, test_acc = common.evaluate(model, device, valid_iterator, criterion)