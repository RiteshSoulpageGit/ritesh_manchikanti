import pandas as pd
from docx import Document
import re
import pandas as pd
import torch
from transformers import BigBirdTokenizer,BigBirdModel,BigBirdForSequenceClassification
from torch import Tensor
import numpy as np
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
from langchain.text_splitter import TokenTextSplitter
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BigBirdConfig
import warnings
warnings.filterwarnings("ignore")
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import classification_report
# from tqdm import tqdm
# from tqdm.notebook import tqdm
from tqdm.auto import tqdm
import math
import matplotlib.pyplot as plt

import torch.multiprocessing as mp 
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend = "nccl",rank = rank, world_size= world_size)



class Trainer:
    def __init__(self,model,tokenizer,train_data,test_loader,train_dataset,test_dataest,optimizer,scheduler,gpu_id,save_every):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.model = DDP(self.model, device_ids = [self.gpu_id],find_unused_parameters=True)

        self.tokenizer = tokenizer
        self.train_data = train_data
        self.train_dataset = train_dataset
        self.test_dataset = test_dataest
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
        self.train_loss_per_epoch = []
        self.val_loss_per_epoch = []
        self.valid_pred = []
        self.EPOCHS = 2
        self.LEARNING_RATE = 0.0000025 ######
        self.BATCH_SIZE = 5
        
    # def _run_batch(self, source, targets):
    #     self.optimizer.zero_grad()
    #     output = self.model(source)
    #     loss = torch.nnCrossEntropyLoss()(output,targets)
    #     loss.backward()
    #     self.optimizer.step()

    def _run_epoch(self, epoch):
        print(f'Epoch: {epoch + 1}')
        self.model.train()
        train_loss = 0
        for step_num, batch_data in enumerate(tqdm(self.train_data, desc='Training')):
            input_ids, att_mask, labels = batch_data["input_ids"].to(self.gpu_id), batch_data["attention_mask"].to(self.gpu_id), batch_data["label"].to(self.gpu_id)
            input_ids = input_ids.to(torch.long)
            att_mask = att_mask.to(torch.long)
            labels = labels.to(torch.long)
            output = self.model(input_ids=input_ids, attention_mask=att_mask, labels=labels)
            loss = output.loss
            train_loss += loss.item()
            self.model.zero_grad()
            loss.backward()
            del loss
            clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            # self._run_batch(input_ids, labels)

        self.train_loss_per_epoch.append(train_loss / (step_num + 1))
        self.model.eval()
        valid_loss = 0
        self.valid_pred = []
        with torch.no_grad():
            for step_num_e, batch_data in enumerate(tqdm(self.test_loader,desc='Validation')):
                # input_ids, att_mask, labels = batch_data["input_ids"].to(device),batch_data["attention_mask"].to(device),batch_data["label"].to(device)
                input_ids, att_mask, labels = batch_data["input_ids"].to(self.gpu_id),batch_data["attention_mask"].to(self.gpu_id),batch_data["label"].to(self.gpu_id)

                # input_ids, att_mask, labels = [data.to(device) for data in batch_data]
                output = self.model(input_ids = input_ids, attention_mask=att_mask, labels= labels)
                # print("logits***:", output["logits"])
                loss = output.loss
                valid_loss += loss.item()
                self.valid_pred.append(np.argmax(output.logits.cpu().detach().numpy(),axis=-1))
        self.val_loss_per_epoch.append(valid_loss / (step_num_e + 1))
        self.valid_pred = np.concatenate(self.valid_pred)
        print("{0}/{1} train loss: {2} ".format(step_num+1, math.ceil(len(self.train_dataset) / self.BATCH_SIZE), train_loss / (step_num + 1)))
        print("{0}/{1} val loss: {2} ".format(step_num_e+1, math.ceil(len(self.test_dataset) / self.BATCH_SIZE), valid_loss / (step_num_e + 1)))          

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        torch.save(ckp,"./model_ddp/checkpoint.pt")
        print (f"Epoch {epoch} | Training checkpoint saved at checkpoint.pt")
    
    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            # if epoch % self.save_every == 0:
            #     self._save_checkpoint(epoch)
    def plot_loss(self):
        epochs = range(1, self.EPOCHS + 1)
        plt.figure(figsize=(8, 8))
        plt.plot(epochs, self.train_loss_per_epoch, label='training loss')
        plt.plot(epochs, self.val_loss_per_epoch, label='validation loss')
        plt.title("Training and Validation Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        # Save the plot as a PNG file
        plt.savefig('./loss_plotBig_bird_new_data.png')

    def show_classification_report(self):
        valid_true = [batch["label"].detach().cpu().numpy() for batch in self.test_loader]
        valid_true = np.concatenate(valid_true)
        print(classification_report(self.valid_pred, valid_true,labels=[0,1,2], target_names= ["level 1","level 2","level 3"]))
        if self.gpu_id == 0:
            self.model.Module.save_pretrained("./savedmodel_multi_gpu_ddp")
            self.tokenizer.save_pretrained("./savedmodel_multi_gpu_ddp/")

def split_tokens_into_smaller_chunks(input_id: Tensor,att_mask: Tensor, chunk_size: int, stride: int, minimal_chunk_length: int):
        input_id_chunks = [input_id[i : i + chunk_size] for i in range(0, len(input_id), stride)]
        mask_chunks = [att_mask[i : i + chunk_size] for i in range(0, len(att_mask), stride)]
        if len(input_id_chunks) > 1:
            # ignore chunks with less than minimal_length number of tokens
            input_id_chunks = [x for x in input_id_chunks if len(x) >= minimal_chunk_length]
            mask_chunks = [x for x in mask_chunks if len(x) >= minimal_chunk_length]
        return input_id_chunks, mask_chunks

def add_special_tokens_at_beginning_and_end(input_id_chunks, mask_chunks) -> None:
        """
        Adds special CLS token (token id = 101) at the beginning.
        Adds SEP token (token id = 102) at the end of each chunk.
        Adds corresponding attention masks equal to 1 (attention mask is boolean).
        """
        for i in range(len(input_id_chunks)):
            # adding CLS (token id 101) and SEP (token id 102) tokens
            input_id_chunks[i] = torch.cat([Tensor([101]), input_id_chunks[i], Tensor([102])])
            # adding attention masks  corresponding to special tokens
            mask_chunks[i] = torch.cat([Tensor([1]), mask_chunks[i], Tensor([1])])

def add_padding_tokens(input_id_chunks, mask_chunks) -> None:
        """Adds padding tokens (token id = 0) at the end to make sure that all chunks have exactly 512 tokens."""
        for i in range(len(input_id_chunks)):
            # get required padding length
            pad_len = 512 - input_id_chunks[i].shape[0]
            # check if tensor length satisfies required chunk size
            if pad_len > 0:
                # if padding length is more than 0, we must add padding
                input_id_chunks[i] = torch.cat([input_id_chunks[i], Tensor([0] * pad_len)])
                mask_chunks[i] = torch.cat([mask_chunks[i], Tensor([0] * pad_len)])


class CustomDataset(Dataset):
        def __init__(self, input_ids, attention_mask, labels_vec):
            self.input_ids = [chunk.long() for chunk in input_ids]
            # print("length in put ids:", len(self.input_ids))
            self.attention_mask = attention_mask
            self.labels_vec = labels_vec

        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, index):
            # print(self.labels_vec[index])
            return {
                'input_ids': self.input_ids[index].squeeze(),
                'attention_mask': self.attention_mask[index].squeeze(),
                'label': self.labels_vec[index]
            }
def stack_tokens_from_all_chunks(input_id_chunks, mask_chunks):
        input_ids = torch.stack(input_id_chunks)
        attention_mask = torch.stack(mask_chunks)
        return input_ids.long(), attention_mask.int()
    
def data_prep(sample_df,tokenizer):
        text_splitter_1 = TokenTextSplitter(chunk_size=300, chunk_overlap=128)
        text_splitter_2 = TokenTextSplitter(chunk_size=150, chunk_overlap=64)
        new_chunk_df = pd.DataFrame()
        total_chunks = []  
        total_att_mask = [] 
        total_fnames = []
        total_labels = []
        total_chunk_counts = []
        label2id = {"Level 1": 0, "Level 2": 1, "Level 3": 2}
        chunks_count = 1
        chunk_length = 2000
        stride = 500
        min_chunk_length = 256
        length_total_labels = 0
        chunk_len_list = []
        token_len_list = []
        for idx in range(sample_df.shape[0]):
        # for idx in range(2):
            x = sample_df["text"].iloc[idx]
            # sentences = re.split(r'(?<=[.!?])\s+', x)
            # long_string = ' '.join(sentences)
            label = label2id[sample_df.iloc[idx]["level"]]
            # chunks = split_text_into_chunks(x, stride, chunk_length, min_chunk_length)
            chunks = text_splitter_1.split_text(x)
            # length_total_labels += len(chunks)
            tokenized_chunks = []
            for chunk in chunks:
                # print("chunk_length",len(chunk.split()))
                chunk_len_list.append(len(chunk.split()))
                tokens = tokenizer(chunk, truncation=True, padding=True, return_tensors="pt")
                if len(tokens['input_ids'][0]) > 512:
                    sub_chunks = text_splitter_2.split_text(chunk)
                    print(len(sub_chunks))
                    for i in sub_chunks:
                        tokens = tokenizer(chunk, truncation=True, padding=True, return_tensors="pt")
                        tokenized_chunks.append(tokens)
                    # print(len(tokens['input_ids'][0]))
                else:
                    token_len_list.append(len(tokens['input_ids'][0]))
                    tokenized_chunks.append(tokens)
            input_id_chunks = []
            mask_chunks = []
            for chunk in tokenized_chunks:
                input_id_chunks.extend(chunk["input_ids"])
                mask_chunks.extend(chunk["attention_mask"])
            add_padding_tokens(input_id_chunks, mask_chunks)
            label_vec = [label] * len(input_id_chunks)
            total_chunks.extend((input_id_chunks))
            total_att_mask.extend((mask_chunks))
            total_labels.extend((label_vec))
        return total_chunks,total_att_mask,total_labels

def load_train_objs():
    Data_csv = pd.read_csv("/home/ubuntu/cat_poc/llms/final_data.csv")
    sample_df = Data_csv.groupby("level").apply(lambda x: x.sample(1))
    tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
    model = BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base', 
                                                        num_labels=3)
    total_chunks,total_att_mask,total_labels = data_prep(sample_df,tokenizer)
    dataset =  CustomDataset(total_chunks,total_att_mask,total_labels)
    train_ratio = 0.8  # 80% for training
    test_ratio = 0.2  # 20% for testing

    num_samples = len(dataset)
    train_size = int(train_ratio * num_samples)
    test_size = num_samples - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=5,
                                        pin_memory=True,
                                        shuffle=False,
                                        sampler = DistributedSampler(train_dataset)
                                        )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=5,
                                            pin_memory=True,
                                            sampler= DistributedSampler(test_dataset)
                                            )
    EPOCHS = 2
    LEARNING_RATE = 0.0000025 ######
    BATCH_SIZE = 5
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.04) #####
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0.04)

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                num_warmup_steps=50, ########
                num_training_steps=len(train_loader)*EPOCHS ) 
    return train_dataset,test_dataset, model,tokenizer, optimizer, scheduler,train_loader,test_loader


def main(rank, world_size,total_epochs, save_every):
    ddp_setup(rank,world_size)
    train_dataset, test_dataset,model, tokenizer,optimizer, scheduler,train_loader,test_loader = load_train_objs()
    print("The length of the training dataset is ",len(train_dataset))
    trainer = Trainer(model,tokenizer, train_loader,test_loader,train_dataset,test_dataset, optimizer,scheduler, rank, save_every)
    trainer.train(total_epochs)
    trainer.plot_loss()
    trainer.show_classification_report()
    print("the jobs are done ending all the processes.")
    destroy_process_group()


if __name__ == "__main__":
    import sys
    total_epochs = 2
    save_every = 1
    # device = 3
    world_size = torch.cuda.device_count()
    mp.spawn(main,args =(world_size,total_epochs,save_every),nprocs=world_size )
    # main(device,total_epochs,save_every)