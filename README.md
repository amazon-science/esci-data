# ESCI Challenge for Improving Product Search - KDD CUP 2022: Baselines 

This is an open source implementation of the baselines presented in the [Amazon Product Search KDD CUP 2022](https://www.aicrowd.com/challenges/esci-challenge-for-improving-product-search).


## Requirements
We launched the baselines experiments creating an environment with Python 3.6 and installing the packages dependencies shown below:
```
aicrowd-cli==0.1.15
numpy==1.19.2
pandas==1.1.5
torch==1.7.1
transformers==4.16.2
scikit-learn==0.24.1
sentence-transformers==2.1.0
```

For installing the dependencies we can launch the following command:
```bash
pip install requirements.txt
```

## Download data

Before to launch the script below, it would be necessary to login in [aicrowd](https://www.aicrowd.com/) using the Python client `aicrowd login`.

The script below downloads all the files for the three tasks using the aicrowd client.

```bash
cd data/
./download-data.sh
```

## Reproduce published results

For a task **K**, we provide the same scripts, one for training the model (and preprocessing the data for tasks 2 and 3): `launch-experiments-taskK.sh`; and a second script for getting the predictions for the public test set using the model trained on the previous step: `launch-predictions-taskK.sh`.

### Task 1 - Query Product Ranking

For task 1, we fine-tuned 3 models one for each `query_locale`.

For `us` locacale we fine-tuned [MS MARCO Cross-Encoders](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2). For `es` and `jp` locales [multilingual MPNet](https://huggingface.co/sentence-transformers/all-mpnet-base-v1). We used the query and title of the product as input for these models.


```bash
cd ranking/
./launch-experiments-task1.sh
./launch-predictions-task1.sh
```

### Task 2 - Multiclass Product Classification

For task 2, we trained a Multilayer perceptron (MLP) classifier whose input is the concatenation of the representations provided by [BERT multilingual base](https://huggingface.co/bert-base-multilingual-uncased) for the query and title of the product.

```bash
cd classification_identification/
./launch-experiments-task2.sh
./launch-predictions-task2.sh
```

### Task 3 - Product Substitute Identification

For task 3, we followed the same approach as in task 2.

```bash
cd classification_identification/
./launch-experiments-task3.sh
./launch-predictions-task3.sh
```

## Results
The following table shows the baseline results obtained through the different public tests of the three tasks.

| Task |  Metric  | Score |
|:----:|:--------:|:-----:|
|    1 | nDCG     | 0.852 |
|    2 | Micro F1 | 0.655 |
|    3 | Micro F1 | 0.780 |

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## Cite

Please cite our paper if you use this dataset for your own research:

```BibTeX
@article{reddy2022shopping,
title={Shopping Queries Dataset: A Large-Scale {ESCI} Benchmark for Improving Product Search},
author={Chandan K. Reddy and Lluís Màrquez and Fran Valero and Nikhil Rao and Hugo Zaragoza and Sambaran Bandyopadhyay and Arnab Biswas and Anlu Xing and Karthik Subbian},
year={2022},
eprint={2206.06588},
archivePrefix={arXiv}
}
```
## License

This project is licensed under the Apache-2.0 License.
