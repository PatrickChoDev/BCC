import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from data.dataset import DCMDataset
from data.loader import dataset_collate
from model.simple_classification import DicomClassifier
from tqdm import tqdm


model = DicomClassifier()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


SPLIT_SIZE=0.5
TRAIN_BATCH_SIZE = 4
LR = 0.001
EPOCHS = 100


optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)


dataset = DCMDataset('Raw Data', 'Raw Data/mammogram.csv', dataset_collate)
print("Dataset size:",len(dataset))
train_dataset, test_dataset = random_split(
    dataset, [int(len(dataset)*SPLIT_SIZE), len(dataset)-int(len(dataset)*SPLIT_SIZE)])
train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)


with torch.enable_grad():
    model.train()
    for epoch in range(EPOCHS):
        with tqdm(train_dataloader) as dataloader:
            for data,label in dataloader:
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                loss = F.binary_cross_entropy_with_logits(output, label)
                del data,label
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                dataloader.set_postfix(loss=loss.item())
        dataloader.set_description(f'Epoch: {epoch}',)
