# ritesh_manchikanti
# Code Repository

This repository contains code for a project that involves training a BigBird model for sequence classification. The code performs various tasks, including data preprocessing, model training, and evaluation.

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- `pandas` library
- `docx` library
- `re` library
- `torch` library
- `transformers` library
- `numpy` library
- `scikit-learn` library
- `matplotlib` library
- `subprocess` module
- `xml.etree.ElementTree` module
- `torch.multiprocessing` module

You can install the required libraries by running the following command:


## Installation

To use this code, follow these steps:

1. Clone the repository:



2. Install the required dependencies (see Prerequisites section).

3. Run the Python script `main.py` to execute the code.

## Usage

The main functionality of the code is encapsulated in the `Trainer` class, which handles model training and evaluation. To use the code, create an instance of the `Trainer` class and pass the required parameters:

```python
model = BigBirdModel(...)
tokenizer = BigBirdTokenizer(...)
train_data = ...
test_loader = ...
train_dataset = ...
test_dataset = ...
optimizer = ...
scheduler = ...
gpu_id = ...
save_every = ...
epochs = ...
lr_rate = ...
batch_size = ...
save_path = ...

trainer = Trainer(model, tokenizer, train_data, test_loader, train_dataset, test_dataset, optimizer, scheduler, gpu_id, save_every, epochs, lr_rate, batch_size, save_path)
trainer.train(max_epochs=10)

