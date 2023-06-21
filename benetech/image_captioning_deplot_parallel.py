import transformers
print(transformers.__version__)

import pandas as pd
from glob import  glob
import json
from tqdm import trange, tqdm
from sklearn.model_selection import StratifiedKFold
import torch
import numpy as np
import random
import os
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm.notebook import tqdm
MAX_PATCHES = 2048

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # Use GPUs 2 and 3

# import torch.multiprocessing as mp
# mp.set_start_method('spawn')

class ImageCaptioningDataset(Dataset):
    def __init__(self, df, processor):
        self.dataset = df
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx, :]
        image = Image.open(row.image_path)
        encoding = self.processor(images=image,
                                  text="Generate underlying data table of the figure below:",
                                  font_path="arial.ttf",
                                  return_tensors="pt",
                                  add_special_tokens=True, max_patches=MAX_PATCHES)
        
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["text"] = row.label
        return encoding
    
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration, AutoProcessor
model = Pix2StructForConditionalGeneration.from_pretrained("./deplot")
processor = AutoProcessor.from_pretrained("./deplot")
model.load_state_dict(torch.load("./weights/deplot_v4/deplot_v4_v4_v4_0.bin"))

def collator(batch):
    new_batch = {"flattened_patches": [], "attention_mask": []}
    texts = [item["text"] for item in batch]
    text_inputs = processor.tokenizer(text=texts,
                                      padding="max_length",
                                      return_tensors="pt",
                                      add_special_tokens=True,
                                      max_length=512,
                                      truncation=True
                                      )

    new_batch["labels"] = text_inputs.input_ids

    for item in batch:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])

    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

    return new_batch

df = pd.read_csv('./datasets/train_with_df_large_cwq2.csv')
print(len(df))
df.reset_index(drop=True, inplace=True)
train_df = df
train_df

class CFG:
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5  # 1.5
    num_warmup_steps = 0.2
    max_input_length = 130
    epochs = 2  # 5
    encoder_lr = 1e-6
    decoder_lr = 1e-6
    min_lr = 0.1e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    weight_decay = 0
    num_fold = 5
    batch_size = 4
    seed = 1006
    num_workers = 2
    device='cuda'
    print_freq = 100

train_dataset = ImageCaptioningDataset(train_df, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=CFG.batch_size, collate_fn=collator, pin_memory=True,
                                  prefetch_factor=40, num_workers=2)

print(train_dataset[2])

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(CFG.seed)

def get_scheduler(cfg, optimizer, num_train_steps):
    cfg.num_warmup_steps = cfg.num_warmup_steps * num_train_steps
    if cfg.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
        )
    elif cfg.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps,
            num_cycles=cfg.num_cycles
        )
    return scheduler

num_train_steps = int(len(train_dataset) / CFG.batch_size * CFG.epochs)

model = torch.nn.DataParallel(model, device_ids=[0,1])  # TODO: adapt to your hardware
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6, weight_decay=2e-7)
scheduler = get_scheduler(CFG, optimizer, num_train_steps)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(device)
model.to(device)
scaler = torch.cuda.amp.GradScaler()
model.train()

import matplotlib.pyplot as plt

loss_file = open("./loss/loss_deplot_v4v4v4v4.txt","w")

for epoch in range(CFG.epochs):
    print("Epoch:", epoch)
    torch.cuda.empty_cache()
    print('outside loop!')
    for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        print(f'inside loop, idx: {idx}')
        labels = batch.pop("labels").to(device)
        flattened_patches = batch.pop("flattened_patches").to(device)
        attention_mask = batch.pop("attention_mask").to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(flattened_patches=flattened_patches,
                            attention_mask=attention_mask,
                            labels=labels)

        loss = outputs.loss.mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if idx % 100 == 0:
            print("Epoch:", epoch, "Loss:", loss.item(), f'lr : {scheduler.get_lr()[0]:.9f} ', sep=' ')
            loss_file.write(f"Epoch: {epoch}, Iteration: {idx}, Loss: {loss.item()}\n")
            loss_file.flush()
        
        print(f'end of iteration: {idx}')

    if (epoch + 1) % 1 == 0:
        torch.save(model.module.state_dict(), f'./weights/deplot_v4/deplot_v4v4v4v4_{epoch}.bin')

loss_file.close()

