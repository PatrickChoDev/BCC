import os,torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from data.dataset import DCMDataset
from data.loader import dataset_collate
from model.simple_classification import DicomClassifier
from tqdm import tqdm


model = DicomClassifier()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



SPLIT_SIZE = 0.5
TRAIN_BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
LR = 0.001
EPOCHS = 100
CHECKPOINT_STEP = 3
CHECKPOINT_DIR = 'checkpoints'

LOAD_CHAECKPOINT = '/home/patrick/Workspace/Research/Cancer classification/checkpoints/epoch-3.pth'


optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

dataset = DCMDataset('Raw Data', 'Raw Data/mammogram.csv', dataset_collate)
print("Dataset size:", len(dataset))
train_dataset, test_dataset = random_split(
    dataset, [int(len(dataset)*SPLIT_SIZE), len(dataset)-int(len(dataset)*SPLIT_SIZE)])
train_dataloader = DataLoader(
    train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

model.to(device)
if LOAD_CHAECKPOINT:
    model.load_state_dict(torch.load(LOAD_CHAECKPOINT))

if CHECKPOINT_STEP and not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

with torch.enable_grad():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        with tqdm(train_dataloader, leave=False) as dataloader:
            for batch_idx, (data, label) in enumerate(dataloader):
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                loss = F.binary_cross_entropy_with_logits(
                    output, label) / ACCUMULATION_STEPS
                total_loss += loss.item()
                del data, label
                loss.backward()
                if ((batch_idx + 1) % ACCUMULATION_STEPS == 0) or (batch_idx + 1 == len(dataloader)):
                    optimizer.step()
                    optimizer.zero_grad()
                dataloader.set_postfix(loss=loss.item())
            scheduler.step()
        dataloader.set_description(f'Epoch: {epoch}',)
        if epoch % CHECKPOINT_STEP == 0 and CHECKPOINT_STEP != 0:
            torch.save(model.state_dict(), f'{CHECKPOINT_DIR}/epoch-{epoch + 1}-{total_loss:.4f}.pth')
        print("Epoch: ", epoch + 1, "\tLoss: ", total_loss)
