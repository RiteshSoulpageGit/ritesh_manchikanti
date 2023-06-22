import pandas as pd
from docx import Document
import docx
import re
import pandas as pd
import torch
from transformers import BigBirdTokenizer,BigBirdModel,BigBirdForSequenceClassification
from torch import Tensor
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain.text_splitter import TokenTextSplitter


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

def read_docx(file_path):
    doc = docx.Document(file_path)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return '\n'.join(text)

def predict_label(file_path, tokenizer, model):
    text_splitter_1 = TokenTextSplitter(chunk_size=300, chunk_overlap=128)
    text_splitter_2 = TokenTextSplitter(chunk_size=150, chunk_overlap=64)

    label2id = {"Level 1": 0, "Level 2": 1, "Level 3": 2}
    id_to_label = {v: k for k, v in label2id.items()}
    true_labels = []
    predicted_labels = []
    text = read_docx(file_path)

    chunks = text_splitter_1.split_text(text)
    tokenized_chunks = []

    for chunk in chunks:
        tokens = tokenizer(chunk, truncation=True, padding=True, return_tensors="pt")
        if len(tokens['input_ids'][0]) > 512:
            sub_chunks = text_splitter_2.split_text(chunk)
            for sub_chunk in sub_chunks:
                tokens = tokenizer(sub_chunk, truncation=True, padding=True, return_tensors="pt")
                tokenized_chunks.append(tokens)
        else:
            tokenized_chunks.append(tokens)

    chunk_predicted_labels = []

    for tokens in tokenized_chunks:
        outputs = model(**tokens)
        predicted_class_index = outputs.logits.argmax().item()
        predicted_class_label = id_to_label[predicted_class_index]
        chunk_predicted_labels.append(predicted_class_label)

    majority_label = max(set(chunk_predicted_labels), key=chunk_predicted_labels.count)

    return majority_label

def main():
    tokenizer = AutoTokenizer.from_pretrained("./ddp/savedmodel_multi_gpu_ddp/")
    model = AutoModelForSequenceClassification.from_pretrained("./ddp/savedmodel_multi_gpu_ddp/")
    file_path = '/home/ubuntu/ritesh_manchikanti/Bigbird/15031-4983-FullBook.docx'
    predicted_label = predict_label(file_path, tokenizer, model)
    print(predicted_label)

if __name__ == "__main__":
    main()