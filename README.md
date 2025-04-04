# Fake News Detection on FakeNewsCorpus and LIAR

This repository contains the code for the project on fake news detection using the FakeNewsCorpus and LIAR datasets. Logistic Regression, Näive Bayes, and DistilBERT models are used for classification. The project is implemented in Python and uses libraries such as scikit-learn, pandas, and transformers.

Sample logs from our final run of the scripts are stored in the `archives` directory.

## Requirements

Python 3.12 was used for this project. There is no guarantee that it will work with other versions. The code has not been tested with any other version of Python.

All dependencies are listed in the `requirements.txt` file. You can install them using pip:

```bash
pip install -r requirements.txt
```

**NOTE 1**: If you are using a CUDA-capable GPU, make sure to install the appropriate version of PyTorch with CUDA support. The `requirements.txt` file includes the CPU version of PyTorch by default. If you want to use the GPU version, you can modify the `requirements.txt` file accordingly. You can find the installation instructions [here](https://pytorch.org/get-started/locally/). 

**NOTE 2**: You should also verify FP16, BF16, and TF32 support for your GPU. The `cuda_checker.py` is not intended to provide an absolute guarantee of compatibility. You can find more information about CUDA compatibility [here](https://pytorch.org/docs/stable/notes/cuda.html#cuda-compatibility).

## How to run and get similar results as our sample logs

0. Download and extract the datasets into the `data` directory. The directory structure should look like this:

```bash
data/
    ├── news_cleaned_2018_02_13.csv # FakeNewsCorpus dataset
    └── liar_test.tsv # LIAR test dataset
```

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies using the command mentioned above.
3. Run `fnc1a.py` to process the FakeNewsCorpus dataset and create the `processed_fakenews.parquet` file. This file will be used for training and evaluating our models.
   - The script will also create a `processed_fakenews.csv` file, which is the same as its parquet version but in CSV format. Just for reference.
   - The script will also convert the original CSV file to parquet format and save it as `news_cleaned_2018_02_13.parquet`. This is done to avoid loading the original CSV file every time we need to process the data. The parquet file is much smaller in size and faster to read.
    - **WARNING**: This script will take a long time to run and use a lot of memory. It is recommended to run it in the background or on a separate machine.
      - The final run took approximately 5 hours on a machine with 32GB of RAM + 16GB of swap and 8 CPU cores (`pandarallel` only used 6 to prevent out-of-memory error).
4. Run `fnc1b.py` to sample 10% of the data from `processed_fakenews.parquet` and split it into training, validation, and test datasets. They will be saved as `sampled_fakenews_<split>.parquet` files. The splits are:
   - `train`: 80% of the sampled data
   - `valid`: 10% of the sampled data
   - `test`: 10% of the sampled data
   - The script will also create a `sampled_fakenews_<split>.csv` file, which is the same as its parquet version but in CSV format. Just for reference.
   - **WARNING**: This script will also take quite long to run and use a lot of VRAM. It is recommended to run it in the background or on a separate machine.
     - The final run took approximately 4 hours on a machine with 8GB of VRAM.
5. Run `fnc4a.py` to process the LIAR test dataset and create the `liar_processed.csv` file. This file will be used for evaluating our models.
6. Run `fnc2.py` to train and evaluate the Logistic Regression and Näive Bayes models on the sampled FakeNewsCorpus dataset and the processed LIAR test dataset. The results are displayed in the terminal only. If you want to save the results to a file, run it as follows:

```bash
python -u src/fnc2.py > <output_file_name>.log 2>&1 &
```

7. Run `fnc3.py` to train and evaluate the DistilBERT model on the sampled FakeNewsCorpus dataset. The steps are saved to `src/results`. In case the training is interrupted, you can resume it by running the script again with `trainer.train(resume_from_checkpoint=True)`. The trained model is saved to `fake_news_bert` directory. Again, the results are displayed in the terminal only.
8. Run `fnc4b.py` to evaluate the DistilBERT model on the processed LIAR test dataset. Again, the results are displayed in the terminal only.

## Extra scripts

- `cuda_checker.py`: A script to check if your GPU supports FP16, BF16, and TF32. It will also check if your GPU is compatible with the CUDA version you have installed. Do not completely rely on this script to check for compatibility. It is just a helper script to give you an idea of your GPU's capabilities.
- `parquet_validator.py`: A script to validate the parquet files created by the `fnc1a.py` and `fnc1b.py`. It will check if the files are valid and if they can be read correctly. It will also check if the data types of the columns are correct. You can trust this one.
