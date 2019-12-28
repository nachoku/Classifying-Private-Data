from models import Combined
import os
import torch
from torch import optim, nn
from torchvision import transforms, datasets
import common

print('Start!!')
#DIRECTORY SETTINGS
os.chdir("..")
DATA_DIR = 'zerr_dataset_cleaned'
SAVE_DIR = 'models'
MODEL_SAVE_PATH = 'base_new_transforms_augmented_batch46_60.pt'


#HYPERPARAMETERS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS=60
BATCH_SIZE = 46
criterion = nn.CrossEntropyLoss()
ADAM_OPTIMISER=True
LEARNING_RATE=0.001



#1. DATA INPUT
train_transforms = transforms.Compose([
                           transforms.Resize((256,256)),
                           #transforms.RandomHorizontalFlip(30),
                           #transforms.RandomRotation(10),
                           #transforms.RandomCrop(256),
                           transforms.ToTensor(),
        #                   transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
                       ])

test_transforms = transforms.Compose([
                           transforms.Resize((256,256)),
                           transforms.ToTensor(),
               #            transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
                       ])

print('Starting Data Import...')

#Directory import
train_data = datasets.ImageFolder(os.path.join(DATA_DIR), train_transforms)
train_data, valid_data = torch.utils.data.random_split(train_data, [int(len(train_data)*0.9), len(train_data) - int(len(train_data)*0.9)])
test_data = datasets.ImageFolder(os.path.join(DATA_DIR), test_transforms)

#Existing Dataset Import
# train_data = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=train_transforms)
# train_data, valid_data = torch.utils.data.random_split(train_data, [int(len(train_data)*0.9), len(train_data) - int(len(train_data)*0.9)])
# test_data = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=test_transforms)


print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')


train_iterator = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
valid_iterator = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE)
test_iterator = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)



#2. ANALYSIS
print('Sending model to device')
print(device)
model=Combined().to(device)
print('Sent..')
for param in model.ocnn.parameters():
    param.requires_grad = False

#Hyperparameters

if(ADAM_OPTIMISER):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
else:
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.5)
#Train
print('Train')
best_valid_loss = float('inf')
for epoch in range(EPOCHS):
#    print('Current Epoch-----:')
#    print(epoch)
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
print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:05.2f}%')
