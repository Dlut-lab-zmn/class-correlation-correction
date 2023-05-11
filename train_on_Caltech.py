# imports
import matplotlib.pyplot as plt
import matplotlib
import joblib
import cv2    #把图片读入到数据集中
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import random
import pretrainedmodels
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
matplotlib.style.use('ggplot')
'''SEED Everything'''
def seed_everything(SEED=10):  #应用不同的种子产生可复现的结果
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True # keep True if all the input have same size.
SEED=42
seed_everything(SEED=SEED)
'''SEED Everything'''



device = 'cuda:4'


epochs = 30
BATCH_SIZE = 64

image_paths = list(paths.list_images('./datasets'))
data = []
labels = []
dic_name2label = {}
for image_path in image_paths:
    label_name = image_path.split(os.path.sep)[-2]
    if label_name not in dic_name2label.keys():
        dic_name2label[label_name] = len(dic_name2label.keys())

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    data.append(image)
    labels.append(dic_name2label[label_name])
# print(data)
# data = np.array(data)
# labels = np.array(labels)
# define transforms
train_transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])
val_transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

# divide the data into train, validation, and test set
(X, x_val , Y, y_val) = train_test_split(data, labels,
                                                    test_size=0.2,
                                                    stratify=labels,
                                                    random_state=SEED)
(x_train, x_test, y_train, y_test) = train_test_split(X, Y,
                                                   test_size=0.25,
                                                   random_state=SEED)
print(f"x_train examples: {len(x_train)}\nx_val examples: {len(x_val)}")

# custom dataset
class ImageDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        data = self.X[i][:]

        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            return (data, self.y[i])
        else:
            return data

train_data = ImageDataset(x_train, y_train, train_transform)
val_data = ImageDataset(x_val, y_val, val_transform)
test_data = ImageDataset(x_test, y_test, val_transform)

trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# the resnet34 model
class ResNet34(nn.Module):
    def __init__(self, pretrained):
        super(ResNet34, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained=None)

        # change the classification layer
        self.l0 = nn.Linear(512, len(dic_name2label.keys()))
        self.dropout = nn.Dropout2d(0.4)
    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        x = self.dropout(x)
        l0 = self.l0(x)
        return l0
model = ResNet34(pretrained=True).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(len(train_data)/BATCH_SIZE) * 10/30, int(len(train_data)/BATCH_SIZE)  * 20 / 30],
                                                    gamma=0.1)
# loss function
criterion = nn.CrossEntropyLoss()


# training function
def fit(model, dataloader, confuse_m, pred_m,target_m):
    print('Training')
    model.train()
    running_loss = 0.0
    running_correct = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):

        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss.item()
        _, preds = torch.max(outputs.data, -1)

        pred_labels_one_hot = torch.zeros( confuse_m.shape[1],confuse_m.shape[1]).unsqueeze(0).repeat(preds.shape[0],1,1)
        pred_labels_one_hot[list(range(preds.shape[0])), target, preds] = 1
        pred_labels_one_hot = torch.sum(pred_labels_one_hot,0).to(preds.device)
        confuse_m += pred_labels_one_hot

        pred_logits_one_hot = torch.zeros( confuse_m.shape[1]).unsqueeze(0).repeat(preds.shape[0],1).to(outputs.device)
        pred_logits_one_hot[list(range(preds.shape[0])), target] = torch.softmax(outputs.data, -1)[list(range(preds.shape[0])),target]
        pred_logits_one_hot = torch.sum(pred_logits_one_hot, 0)
        pred_m += pred_logits_one_hot

        rel_labels_one_hot = torch.zeros( confuse_m.shape[1]).unsqueeze(0).repeat(preds.shape[0],1).to(preds.device)
        rel_labels_one_hot[list(range(preds.shape[0])), target] = 1
        rel_labels_one_hot = torch.sum(rel_labels_one_hot,0)
        target_m += rel_labels_one_hot

        running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    loss = running_loss/len(dataloader.dataset)
    accuracy = 100. * running_correct/len(dataloader.dataset)

    print(f"Train Loss: {loss:.4f}, Train Acc: {accuracy:.2f}")

    return loss, accuracy, confuse_m, pred_m, target_m

#validation function
def validate(model, dataloader, confuse_m, pred_m, target_m):
    print('Validating')
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_correct2 = 0
    running_correct3 = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            pred_soft = torch.softmax(outputs.data, -1)
            beta2 = 0
            beta = 1
            running_column_prob = (torch.diag(torch.sum(confuse_m,0)) * beta2  + confuse_m*beta) / (torch.sum(confuse_m,0)*( beta2+ beta))
            running_pred_column_relation_probs = torch.sum( (pred_m/target_m)* running_column_prob,-1)
            fg_relation_logits = pred_soft.unsqueeze(1).repeat(1, running_column_prob.shape[0],1)
            fg_pred_column_relation_probs = torch.sum(fg_relation_logits * running_column_prob,-1)

            relation_probs_norm = fg_pred_column_relation_probs/ (running_pred_column_relation_probs + 1e-5) 
            relation_probs_norm2 = pred_soft/ ((pred_m/target_m) + 1e-5) 

            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            _, preds2 = torch.max(relation_probs_norm, 1)
            _, preds3 = torch.max(relation_probs_norm2, 1)
            running_correct += (preds ==target).sum().item()
            running_correct2 += (preds2 ==target).sum().item()
            running_correct3 += (preds3 ==target).sum().item()

        loss = running_loss/len(dataloader.dataset)
        accuracy = 100. * running_correct/len(dataloader.dataset)
        accuracy2 = 100. * running_correct2/len(dataloader.dataset)
        accuracy3 = 100. * running_correct3/len(dataloader.dataset)
        print(f'Val Loss: {loss:.4f}, Val Acc: {accuracy:.2f},  Val Acc2: {accuracy2:.2f},  Val Acc3: {accuracy3:.2f}')

        return loss, accuracy, accuracy2, accuracy3

def test(model, dataloader, confuse_m, pred_m, target_m):
    print('Testing')
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_correct2 = 0
    running_correct3 = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            pred_soft = torch.softmax(outputs.data, -1)
            beta2 = 0
            beta = 1
            running_column_prob = (torch.diag(torch.sum(confuse_m,0)) * beta2  + confuse_m*beta) / (torch.sum(confuse_m,0)*( beta2+ beta))
            running_pred_column_relation_probs = torch.sum( (pred_m/target_m)* running_column_prob,-1)
            fg_relation_logits = pred_soft.unsqueeze(1).repeat(1, running_column_prob.shape[0],1)
            fg_pred_column_relation_probs = torch.sum(fg_relation_logits * running_column_prob,-1)

            relation_probs_norm = fg_pred_column_relation_probs/ (running_pred_column_relation_probs + 1e-5) 
            relation_probs_norm2 = pred_soft/ ((pred_m/target_m) + 1e-5) 

            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            _, preds2 = torch.max(relation_probs_norm, 1)
            _, preds3 = torch.max(relation_probs_norm2, 1)
            running_correct += (preds ==target).sum().item()
            running_correct2 += (preds2 ==target).sum().item()
            running_correct3 += (preds3 ==target).sum().item()

        loss = running_loss/len(dataloader.dataset)
        accuracy = 100. * running_correct/len(dataloader.dataset)
        accuracy2 = 100. * running_correct2/len(dataloader.dataset)
        accuracy3 = 100. * running_correct3/len(dataloader.dataset)
        print(f'Val Loss: {loss:.4f}, Val Acc: {accuracy:.2f},  Val Acc2: {accuracy2:.2f},  Val Acc3: {accuracy3:.2f}')

        return loss, accuracy, accuracy2, accuracy3


train_loss , train_accuracy = [], []
val_loss , val_accuracy, val_accuracy2, val_accuracy3 = [], [], [], []
test_loss , test_accuracy, test_accuracy2, test_accuracy3 = [], [], [], []
print(f"Training on {len(train_data)} examples, validating on {len(val_data)} examples...")
start = time.time()
confuse_m = torch.zeros([len(dic_name2label), len(dic_name2label)]).to(device)
pred_m = torch.zeros([len(dic_name2label)]).to(device)
target_m = torch.zeros([len(dic_name2label)]).to(device) + 1
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_accuracy, confuse_m, pred_m, target_m = fit(model, trainloader, confuse_m, pred_m, target_m)
    val_epoch_loss, val_epoch_accuracy, val_epoch_accuracy2, val_epoch_accuracy3 = validate(model, valloader, confuse_m, pred_m, target_m)
    test_epoch_loss, test_epoch_accuracy, test_epoch_accuracy2, test_epoch_accuracy3 = test(model, testloader, confuse_m, pred_m, target_m)
    # val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
    val_accuracy2.append(val_epoch_accuracy2)
    val_accuracy3.append(val_epoch_accuracy3)
    test_accuracy.append(test_epoch_accuracy)
    test_accuracy2.append(test_epoch_accuracy2)
    test_accuracy3.append(test_epoch_accuracy3)
print(val_accuracy)
print(val_accuracy2)
print(val_accuracy3)
print(test_accuracy)
print(test_accuracy2)
print(test_accuracy3)

end = time.time()
print((end-start)/60, 'minutes')
torch.save(model.state_dict(), f"./output/resnet34_epochs{epochs}.pth")
# accuracy plots

x = np.arange(22) + 8

plt.figure(figsize=(10, 7))
# plt.plot(train_accuracy, color='green', label='ORI train accuracy')
plt.plot(x, val_accuracy[8:], color='blue', label='ORI(VAL)')
plt.plot(x, val_accuracy2[8:], color='red', label='$F3C\dagger(\alpha = 0)$(VAL)')
plt.plot(x, val_accuracy3[8:], color='orange', label='DLFE(VAL)')
plt.plot(x, test_accuracy[8:], '-.',color='blue', label='ORI(TEST)')
plt.plot(x, test_accuracy2[8:], '-.',color='red', label='$F3C\dagger(\alpha = 0)$(TEST)')
plt.plot(x, test_accuracy3[8:], '-.',color='orange', label='DLFE(TEST)')

plt.xlabel('Epochs')
plt.ylabel('Classification Accuracy')
plt.legend()
plt.savefig('./output/plots/accuracy4.png')
# loss plots
