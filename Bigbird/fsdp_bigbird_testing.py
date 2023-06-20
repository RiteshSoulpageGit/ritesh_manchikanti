# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
# needs Float
# model = model.half()
# mixed precision set to auto a9pplyaaaa4R  ]-090-][]\09 AZXD2=\]]
# GPT2 with T5-Block
# from transformers import BigBirdConfig, BigBirdModel
# from transformers.models.big_bird.modeling_big_bird import BigBirdBlock


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
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
 #checkpoint_wrapper,
 CheckpointImpl)
# apply_activation_checkpointing_wrapper)

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
    
)


#----------------data-preparation---------------
# train_data = pd.read_parquet('/home/ubuntu/working_directory/Bert_experimentation/80_train_new_modified.parqa


def split_text_into_chunks(text, stride, chunk_length, min_chunk_length):
    # Split the text into sentences
    sentences = re.split(r"(?<=\.|\?|\!)\s", text)
    
    chunks = []
    current_chunk = ""
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        # If adding the current sentence to the chunk will exceed the chunk length, start a new chunk
        if current_length + sentence_length > chunk_length:
            if current_length >= min_chunk_length:
                # last_element = my_list[-1]
                # sub_sentences = chunks[-1]
                if chunks:
                    sub_sentences = re.split(r"(?<=\;|\,|\.|\?|\!)\s", chunks[-1])
                else:
                    sub_sentences = re.split(r"(?<=\;|\,|\.|\?|\!)\s", sentence)

                sub_sentences = re.split(r"(?<=\;|\,|\.|\?|\!)\s", sentence)
                for a, sub_sentence in enumerate(sub_sentences):
                    sub_sentence_length = len(sub_sentence)
                    if current_length + sub_sentence_length > chunk_length:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                        current_length = 0
                    current_chunk += sub_sentence + " "
                    current_length += sub_sentence_length
                    # current_length += len(sentence)
                    # if current_length >= stride:
                    #     remove_index = a + 1
                    #     break

                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_length = 0
        current_chunk += sentence + " "
        current_length += sentence_length
    if current_length >= min_chunk_length:
        chunks.append(current_chunk.strip())
    return chunks

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



# def process_data(sample_df, tokenizer, label2id, chunk_length=510, stride=386, min_chunk_length=256):
#     def split_text_into_chunks(text, stride, chunk_length, min_chunk_length):
#         sentences = re.split(r'(?<=[.!?])\s+', text)
#         long_string = ' '.join(sentences)
#         chunks = []
#         current_chunk = ""
#         current_length = 0

#         for sentence in sentences:
#             sentence_length = len(sentence)
#             if current_length + sentence_length > chunk_length:
#                 if current_length >= min_chunk_length:
#                     chunks.append(current_chunk.strip())
#                 current_chunk = ""
#                 current_length = 0
#             current_chunk += sentence + " "
#             current_length += sentence_length

#         if current_length >= min_chunk_length:
#             chunks.append(current_chunk.strip())

#         return chunks

#     def add_special_tokens_at_beginning_and_end(input_ids, attention_masks):
#         input_ids[:, 0] = tokenizer.cls_token_id
#         input_ids[:, -1] = tokenizer.sep_token_id
#         attention_masks[:, :2] = 1
#         attention_masks[:, -1] = 1

#     def add_padding_tokens(input_ids, attention_masks, max_length):
#         input_ids = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=tokenizer.pad_token_id)
#         attention_masks = F.pad(attention_masks, (0, max_length - attention_masks.shape[1]), value=0)
#         return input_ids, attention_masks

#     new_chunk_df = pd.DataFrame()
#     total_chunks = []
#     total_att_mask = []
#     total_labels = []

#     # label_encoder = OneHotEncoder(sparse=False, categories="auto")
#     # label_ids = label_encoder.fit_transform(label2id.values())

#     for idx in range(sample_df.shape[0]):
#         x = sample_df["text"].iloc[idx]
#         label = sample_df.iloc[idx]["level"]
#         chunks = split_text_into_chunks(x, stride, chunk_length, min_chunk_length)
#         tokenized_chunks = []
#         for chunk in chunks:
#             tokens = tokenizer(chunk, truncation=True, padding=True, return_tensors="pt")
#             tokenized_chunks.append(tokens)

#         input_id_chunks = []
#         mask_chunks = []
#         for chunk in tokenized_chunks:
#             input_id_chunks.extend(chunk["input_ids"])
#             mask_chunks.extend(chunk["attention_mask"])

#         add_special_tokens_at_beginning_and_end(input_id_chunks, mask_chunks)
#         padded_input_id_chunks, padded_mask_chunks = add_padding_tokens(input_id_chunks, mask_chunks, max_chunk_length)

#         total_chunks.extend(padded_input_id_chunks)
#         total_att_mask.extend(padded_mask_chunks)
#         total_labels.extend([label] * len(padded_input_id_chunks))

#     total_chunks = torch.stack(total_chunks)
#     total_att_mask = torch.stack(total_att_mask)
#     total_labels = torch.tensor(total_labels)

#     return total_chunks, total_att_mask, total_labels

def process_text_chunks(sample_df, tokenizer, label2id, chunk_length=510, stride=386, min_chunk_length=256, max_chunk_length=512):
    from sklearn.preprocessing import OneHotEncoder
    import torch
    import torch.nn.functional as F
    from langchain.text_splitter import TokenTextSplitter
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
        return total_chunks, total_att_mask, total_labels


# Usage example



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

def split_train_test(dataset, train_ratio=0.8, test_ratio=0.2):
    num_samples = len(dataset)
    train_size = int(train_ratio * num_samples)
    test_size = num_samples - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset
# Usage example


                                          
#-----------------Data-preparatin-ended------------

#-----------------Model-initilization-started------------

#-----------------Model-initilization-ended------------




def setup():
    # initialize the process group
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()

def setup_model():

    tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
    model = BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base', 
                                                         num_labels=3)

    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # model = GPT2ForSequenceClassification.from_pretrained(model_name, problem_type="multi_label_classification", num_labels = 3)
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     model.resize_token_embeddings(len(tokenizer))
    #     model.config.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    return model, tokenizer


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run

# def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
#     model.train()
#     local_rank = int(os.environ['LOCAL_RANK'])
#     fsdp_loss = torch.zeros(2).to(local_rank)

#     if sampler:
#         sampler.set_epoch(epoch)
#     if rank==0:
#         inner_pbar = tqdm.tqdm(
#             range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
#         )
#     for batch_data in train_loader:
#         input_ids, att_mask, labels = batch_data["input_ids"].to(local_rank),batch_data["attention_mask"].to(local_rank),batch_data["label"].to(local_rank)
#         # batch,label = batch
#         # for key in batch.keys():
#         #     batch[key] = batch[key].to(local_rank)
#         # label = label.to(local_rank) 

#         optimizer.zero_grad()
#         # print("Model again:", model)
#         output = model(input_ids = input_ids, attention_mask=att_mask, labels= labels)
#         loss = output["loss"]

#         loss.backward()
#         optimizer.step()
#         fsdp_loss[0] += loss.item()
#         fsdp_loss[1] += len(batch_data)
#         if rank==0:
#             inner_pbar.update(1)

#     dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
#     train_accuracy = fsdp_loss[0] / fsdp_loss[1]


#     if rank == 0:
#         inner_pbar.close()
#         print(
#                 f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
#             )
#     return train_accuracy



# def train_and_evaluate(model, train_loader, test_loader, optimizer, scheduler, EPOCHS,rank):
#     train_loss_per_epoch = []
#     val_loss_per_epoch = []

#     model = model.to(device)
#     if rank==0:
#         inner_pbar = tqdm.tqdm(
#             range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
#         )
#     for epoch_num in range(EPOCHS):
#         print('Epoch: ', epoch_num + 1)
#         '''
#         Training
#         '''
#         model.train()
#         train_loss = 0
#         for step_num, batch_data in enumerate(tqdm(train_loader, desc='Training')):
#             input_ids, att_mask, labels = batch_data["input_ids"].to(device), batch_data["attention_mask"].to(device), batch_data["label"].to(device)
#             input_ids = input_ids.to(torch.long).to(device)
#             att_mask = att_mask.to(torch.long).to(device)
#             labels = labels.to(torch.long).to(device)

#             output = model(input_ids=input_ids, attention_mask=att_mask, labels=labels)
#             loss = output.loss
#             train_loss += loss.item()

#             model.zero_grad()
#             loss.backward()
#             del loss

#             clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
#             optimizer.step()
#             scheduler.step()

#         train_loss_per_epoch.append(train_loss / (step_num + 1))

#         '''
#         Validation
#         '''
#         model.eval()
#         valid_loss = 0
#         valid_pred = []
#         with torch.no_grad():
#             for step_num_e, batch_data in enumerate(tqdm(test_loader, desc='Validation')):
#                 input_ids, att_mask, labels = batch_data["input_ids"].to(device), batch_data["attention_mask"].to(device), batch_data["label"].to(device)
#                 output = model(input_ids=input_ids, attention_mask=att_mask, labels=labels)

#                 loss = output.loss
#                 valid_loss += loss.item()

#                 valid_pred.append(np.argmax(output.logits.cpu().detach().numpy(), axis=-1))

#         val_loss_per_epoch.append(valid_loss / (step_num_e + 1))
#         valid_pred = np.concatenate(valid_pred)
#         print("{0}/{1} train loss: {2} ".format(step_num + 1, math.ceil(len(train_dataset) / BATCH_SIZE), train_loss / (step_num + 1)))
#         print("{0}/{1} val loss: {2} ".format(step_num_e + 1, math.ceil(len(test_dataset) / BATCH_SIZE), valid_loss / (step_num_e + 1)))

#     return train_loss_per_epoch, val_loss_per_epoch

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

def train_and_evaluate(model, train_loader, test_loader, optimizer, scheduler, EPOCHS, rank):
    train_loss_per_epoch = []
    val_loss_per_epoch = []

    # model = model.to(device)
    fsdp_loss = torch.zeros(2).to(rank)

    if rank == 0:
        inner_pbar = tqdm(range(len(train_loader)), colour="blue", desc="r0 Training Epoch")

    for epoch_num in range(EPOCHS):
        print('Epoch: ', epoch_num + 1)
        '''
        Training
        '''
        model.train()
        train_loss = 0

        for step_num, batch_data in enumerate(train_loader):
            input_ids, att_mask, labels = batch_data["input_ids"].to(rank), batch_data["attention_mask"].to(rank), batch_data["label"].to(rank)
            input_ids = input_ids.to(torch.long).to(rank)
            att_mask = att_mask.to(torch.long).to(rank)
            labels = labels.to(torch.long).to(rank)

            optimizer.zero_grad()
            output = model(input_ids=input_ids, attention_mask=att_mask, labels=labels)
            loss = output.loss

            loss.backward()
            clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            optimizer.step()

            fsdp_loss[0] += loss.item()
            fsdp_loss[1] += len(batch_data)

            if rank == 0:
                inner_pbar.update(1)

        dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
        train_accuracy = fsdp_loss[0] / fsdp_loss[1]

        if rank == 0:
            inner_pbar.close()
            print(f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}")

        '''
        Validation
        '''
        model.eval()
        valid_loss = 0
        valid_pred = []

        if rank == 0:
            inner_pbar = tqdm(range(len(test_loader)), colour="blue", desc="r0 Validation Epoch")

        for step_num_e, batch_data in enumerate(test_loader):
            input_ids, att_mask, labels = batch_data["input_ids"].to(rank), batch_data["attention_mask"].to(rank), batch_data["label"].to(rank)
            output = model(input_ids=input_ids, attention_mask=att_mask, labels=labels)

            loss = output.loss
            valid_loss += loss.item()

            valid_pred.append(np.argmax(output.logits.cpu().detach().numpy(), axis=-1))

        val_loss_per_epoch.append(valid_loss / (step_num_e + 1))
        valid_pred = np.concatenate(valid_pred)

        if rank == 0:
            inner_pbar.close()
            print("{0}/{1} train loss: {2} ".format(step_num + 1, math.ceil(len(train_dataset) / BATCH_SIZE), train_loss / (step_num + 1)))
            print("{0}/{1} val loss: {2} ".format(step_num_e + 1, math.ceil(len(test_dataset) / BATCH_SIZE), valid_loss / (step_num_e + 1)))

    return train_loss_per_epoch, val_loss_per_epoch


label2id = {"Level 1": 0, "Level 2": 1, "Level 3": 2} # ssssssssssssss

def fsdp_main(args):

    model, tokenizer = setup_model()

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    print("No of gpus:", world_size)

    Data_csv = pd.read_csv("/home/ubuntu/cat_poc/llms/final_data.csv")
    # Data_csv
    level_counts = Data_csv['level'].value_counts()
    print(level_counts)
    sample_df = Data_csv.groupby("level").apply(lambda x: x.sample(2))
    
    total_chunks, total_att_mask, total_labels = process_text_chunks(sample_df, tokenizer, label2id)

    dataset =  CustomDataset(total_chunks,total_att_mask,total_labels)
    train_dataset, test_dataset = split_train_test(dataset, train_ratio=0.8, test_ratio=0.2)


    # train_dataset = CustomDataset(train_data, label2id, tokenizer)
    # test_dataset= CustomDataset(val_df, label2id, tokenizer)

    print("*"*50)
    print("Size of train dataset: ", len(train_dataset))
    print("Size of Validation dataset: ", len(test_dataset))

  

    sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size, shuffle= True)

    setup()

    BATCH_SIZE = 5
    # train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    # test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}

    train_kwargs = {'batch_size': BATCH_SIZE, 'sampler': sampler1,'pin_memory':True}
    test_kwargs = {'batch_size': BATCH_SIZE, 'sampler': sampler2,'pin_memory':True}
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    }
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)


    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                        batch_size=5,
    #                                        pin_memory=True,
    #                                        shuffle=True, 
    #                                        )
    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #                                         batch_size=5,
    #                                         pin_memory=True,
    #                                         )

    

    # t5_auto_wrap_policy = functools.partial(
    #     transformer_auto_wrap_policy,
    #     transformer_layer_cls={
    #         GPT2Block,
    #     },
    # )

    t5_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            BigBirdAttention,
        },
    )
    # sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP #for Zero2 and FULL_SHARD for Zero3
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD  #for Zero2 and FULL_SHARD for Zero3
    torch.cuda.set_device(local_rank)


    #init_start_event = torch.cuda.Event(enable_timing=True)
    #init_end_event = torch.cuda.Event(enable_timing=True)

    #init_start_event.record()

    bf16_ready = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and LooseVersion(torch.version.cuda) >= "11.0"
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    )

    if bf16_ready:
        print("BF16 supported")
        mp_policy = bf16_ready
    else:
        print("BF16 not supported")
        mp_policy = None # defaults to fp32

    # model is on CPU before input to FSDP
    model = FSDP(model,
        auto_wrap_policy=t5_auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device())

    EPOCHS = 5
    LEARNING_RATE = 0.0000025 ######
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.04) #####
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0.04)

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                num_warmup_steps=50, ########
                num_training_steps=len(train_loader)*EPOCHS )   

    # optimizer = optim.AdamW(model.parameters(), lr=args.lr,weight_decay=0.45)
    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma, )
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    best_val_loss = float("inf")
    curr_val_loss = float("inf")
    file_save_name = "./bert-XL-model-"

    if rank == 0:
        time_of_run = get_date_of_run()
        dur = []
        train_acc_tracking = []
        val_acc_tracking = []
        training_start_time = time.time()

    if rank == 0 and args.track_memory:
        mem_alloc_tracker = []
        mem_reserved_tracker = []

    # for epoch in range(1, EPOCHS + 1):
    #     t0 = time.time()
    train_loss_per_epoch, val_loss_per_epoch = train_and_evaluate(model, train_loader, test_loader, optimizer, scheduler, EPOCHS, rank)

        # train_accuracy = train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)

        # if args.run_validation:
        #     curr_val_loss,acc,report,   confusion_mat = validation(model, rank, world_size, val_loader)
        # scheduler.step(train_accuracy)

        # if rank == 0:

        #     print(f"--> epoch {epoch} completed...entering save and stats zone")

        #     dur.append(time.time() - t0)
        #     train_acc_tracking.append(train_accuracy.item())

        #     if args.run_validation:
        #         val_acc_tracking.append(curr_val_loss.item())
                
        #     print(f"completed save and stats zone...")


        # if args.save_model and curr_val_loss < best_val_loss:

        #     # save
        #     if rank == 0:
        #         print(f"--> entering save model state")

        #     save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        #     with FSDP.state_dict_type(
        #         model, StateDictType.FULL_STATE_DICT, save_policy
        #     ):
        #         cpu_state = model.state_dict()
        #     print(f"saving process: rank {rank}  done w state_dict")


        #     if rank == 0:
        #         print(f"--> saving model ...")
        #         currEpoch = (
        #             "-" + str(epoch) + "-" + str(round(curr_val_loss.item(), 4)) 
        #         )
        #         print(f"--> attempting to save model prefix {currEpoch}")
        #         save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
        #         print(f"--> saving as model name {save_name}")

        #         model.save_pretrained(f'/home/ubuntu/working_directory/experimentation/Final_GPT_FINAL_REport_model_{currEpoch}/')


        #         # torch.save(cpu_state, save_name)

        # if curr_val_loss < best_val_loss:

        #     best_val_loss = curr_val_loss
        #     if rank==0:
        #         print(f"-->>>> New Val Loss Record: {best_val_loss}")

    dist.barrier()
    cleanup()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch T5 FSDP Example')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=2, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 3)')  
    parser.add_argument('--lr', type=float, default=0.00005 , metavar='LR',
                        help='learning rate (default: .002)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--track_memory', action='store_false', default=True,
                        help='track the gpu memory')
    parser.add_argument('--run_validation', action='store_false', default=True,
                        help='running the validation')
    parser.add_argument('--save-model', action='store_false', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    fsdp_main(args)





    # model.save_pretrained('/home/ubuntu/working_directory/new_model_weight_300/')
     
