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

Usage

# Training

To train the model, run the following command:

``` python your_script.py --mode train --total_epochs X --save_every Y --batch_size Z --lr_rate LR --path /path/to/data.csv --model_save_path /path/to/save/models ```


- `total_epochs:` Total number of epochs. It specifies the number of times the model will iterate over the entire dataset during training.
- `save_every:` Save model every N epochs. It determines how frequently the model should be saved during training. The model will be saved every N epochs specified.
- `batch_size:` Batch size. It indicates the number of samples that will be propagated through the network at once during training.
- `lr_rate:` Learning rate. It represents the step size at which the model learns. It controls the amount by which the model's parameters are updated during training.
- `path:` Data file path. It specifies the location of the data file to be used for training.
- `model_save_path:` Model save path. It determines the directory where the trained model will be saved.
Testing
To perform testing and inference, run the following command:

``` python your_script.py --mode test --tokenizer_path /path/to/tokenizer --file_path /path/to/file ```
- `tokenizer_path /path/to/tokenizer`: Specify the path to the tokenizer used for testing.
- `file_path /path/to/file`: Specify the path to the file for inference.

Note: Make sure to replace /path/to/data.csv, /path/to/save/models, /path/to/tokenizer, and /path/to/file with the actual file paths on your system.



The main functionality of the code is encapsulated in the `Trainer` class, which handles model training and evaluation. To use the code, create an instance of the `Trainer` class and pass the required parameters:

```python
model = BigBirdModel(...)
tokenizer = BigBirdTokenizer(...)
gpu_id = ...
epochs = ...
lr_rate = ...
batch_size = ...
save_path = ...

trainer = Trainer(model, tokenizer, train_data, test_loader, train_dataset, test_dataset, optimizer, scheduler, gpu_id, save_every, epochs, lr_rate, batch_size, save_path)
trainer.train(max_epochs=10)

