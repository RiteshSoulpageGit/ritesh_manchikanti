import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn.functional as F
from langchain.text_splitter import TokenTextSplitter
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.models.big_bird.modeling_big_bird import BigBirdAttention
import pandas as pd
from docx import Document
import re
import pandas as pd
import torch
from transformers import BigBirdTokenizer,BigBirdModel,BigBirdForSequenceClassification
from torch import Tensor
import numpy as np
from collections import Counter
# from typing import list
import numpy as np
import math
from torch.nn.utils import clip_grad_norm_
from tqdm.notebook import tqdm

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, GPT2TokenizerFast
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


from sklearn.metrics import accuracy_score, classification_report

import functools
from torch.utils.data import Dataset 
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import BertModel, AutoConfig

from transformers.models.t5.modeling_t5 import T5Block
from transformers.models.bert.modeling_bert import BertSelfAttention,BertAttention,BertLayer,BertModel
from transformers import GPT2Tokenizer, GPT2Model
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AdamW
import numpy as np
from collections import Counter
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from functools import partial
from torch.utils.data import DataLoader
from pathlib import Path
# from summarization_dataset import *
from transformers.models.t5.modeling_t5 import T5Block
from typing import Type
import time
import tqdm
from datetime import datetime
import pandas as pd
import re
import torch
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split


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

def stack_tokens_from_all_chunks(input_id_chunks, mask_chunks):
    input_ids = torch.stack(input_id_chunks)
    attention_mask = torch.stack(mask_chunks)
    return input_ids.long(), attention_mask.int()

import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
import math
import torch.distributed as dist
import torch.nn as nn

def train(rank, world_size, model, train_loader, test_loader, optimizer, scheduler, device, EPOCHS):
    # Initialize the distributed environment
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Create model and move it to the GPU
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    train_loss_per_epoch = []
    val_loss_per_epoch = []

    for epoch_num in range(EPOCHS):
        print('Epoch: ', epoch_num + 1)
        '''
        Training
        '''
        model.train()
        train_loss = 0
        for step_num, batch_data in enumerate(tqdm(train_loader, desc='Training')):
            input_ids, att_mask, labels = batch_data["input_ids"].to(device), batch_data["attention_mask"].to(device), batch_data["label"].to(device)
            input_ids = input_ids.to(torch.long).to(device)
            att_mask = att_mask.to(torch.long).to(device)
            labels = labels.to(torch.long).to(device)
            output = model(input_ids=input_ids, attention_mask=att_mask, labels=labels)
            loss = output.loss
            train_loss += loss.item()

            model.zero_grad()
            loss.backward()
            del loss

            clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        train_loss_per_epoch.append(train_loss / (step_num + 1))

        '''
        Validation
        '''
        model.eval()
        valid_loss = 0
        valid_pred = []
        with torch.no_grad():
            for step_num_e, batch_data in enumerate(tqdm(test_loader, desc='Validation')):
                input_ids, att_mask, labels = batch_data["input_ids"].to(device), batch_data["attention_mask"].to(device), batch_data["label"].to(device)
                output = model(input_ids=input_ids, attention_mask=att_mask, labels=labels)
                loss = output.loss
                valid_loss += loss.item()

                valid_pred.append(np.argmax(output.logits.cpu().detach().numpy(), axis=-1))

        val_loss_per_epoch.append(valid_loss / (step_num_e + 1))
        valid_pred = np.concatenate(valid_pred)
        print("{0}/{1} train loss: {2} ".format(step_num + 1, math.ceil(len(train_dataset) / BATCH_SIZE),
                                                train_loss / (step_num + 1)))
        print("{0}/{1} val loss: {2} ".format(step_num_e + 1, math.ceil(len(test_dataset) / BATCH_SIZE),
                                              valid_loss / (step_num_e + 1)))

    # Clean up
    dist.destroy_process_group()

    return train_loss_per_epoch, val_loss_per_epoch



def demo_basic():
    # Data_csv = pd.read_csv("/home/ubuntu/working_directory/Bert_experimentation/new_dataframe_50000_100000.csv")
    Data_csv = pd.read_csv("/home/ubuntu/cat_poc/llms/final_data.csv")
    # Data_csv.drop('Unnamed: 0', axis=1, inplace=True)
    # Data_csv
    level_counts = Data_csv['level'].value_counts()
    print(level_counts)
    sample_df = Data_csv.groupby("level").apply(lambda x: x.sample(60))



    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()

    tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
    model = BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base', 
                                                            num_labels=3).to(device_id)

    ddp_model = DDP(model, device_ids=[device_id])

    ##### start of Data Prep ######
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

    ####end of Data Prep ######
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=5,
                                           pin_memory=True,
                                           shuffle=True, 
                                           )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=5,
                                            pin_memory=True,
                                            )


    EPOCHS = 5
    LEARNING_RATE = 0.0000025 ######
    BATCH_SIZE = 5
    optimizer = AdamW(ddp_model.parameters(), lr=LEARNING_RATE, weight_decay=0.04) #####

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                num_warmup_steps=50, ########
                num_training_steps=len(train_loader)*EPOCHS )   
    
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    num_processes = num_gpus
    mp.spawn(train, args=(num_processes, model, train_loader, test_loader, optimizer, scheduler, device, EPOCHS), nprocs=num_processes)

                

if __name__ == "__main__":

    demo_basic()