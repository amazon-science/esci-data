# Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search

## Introduction

We introduce the “Shopping Queries Data Set”, a large dataset of difficult search queries, released with the aim of fostering research in the area of semantic matching of queries and products. For each query, the dataset provides a list of up to 40 potentially relevant results, together with ESCI relevance judgements (Exact, Substitute, Complement, Irrelevant) indicating the relevance of the product to the query. Each query-product pair is accompanied by additional information. The dataset is multilingual, as it contains queries in English, Japanese, and Spanish.


The primary objective of releasing this dataset is to create a benchmark for building new ranking strategies and simultaneously identifying interesting categories of results (i.e., substitutes) that can be used to improve the customer experience when searching for products. The three different tasks that are studied in the literature (see https://amazonkddcup.github.io/) using this Shopping Queries Dataset are:


**Task 1 - Query-Product Ranking**: Given a user specified query and a list of matched products, the goal of this task is to rank the products so that the relevant products are ranked above the non-relevant ones.


**Task 2 - Multi-class Product Classification**: Given a query and a result list of products retrieved for this query, the goal of this task is to classify each product as being an Exact, Substitute, Complement, or Irrelevant match for the query.


**Task 3 - Product Substitute Identification**: This task will measure the ability of the systems to identify the substitute products in the list of results for a given query.

## Dataset

We provide two different versions of the data set. One for task 1 which is reduced version in terms of number of examples and ones for tasks 2 and 3 which is a larger.

The training data set contain a list of query-result pairs with annotated E/S/C/I labels. The data is **multilingual** and it includes queries from **English**, **Japanese**, and **Spanish** languages. The examples in the data set have the following fields: `example_id`, `query`, `query_id`, `product_id`, `product_locale`, `esci_label`, `small_version`, `large_version`, `split`, `product_title`, `product_description`, `product_bullet_point`, `product_brand`, `product_color` and  `source`

The Shopping Queries Data Set is a large-scale manually annotated data set composed of challenging customer queries.

There are 2 versions of the dataset. The reduced version of the data set contains `48,300 unique queries` and `1,118,011 rows` corresponding each to a `<query, item>` judgement. The larger version of the data set contains `130,652 unique queries` and `2,621,738 judgements`. The reduced version of the data accounts for queries that are deemed to be **“easy”**, and hence filtered out. The data is stratified by queries in two splits train, and test.

A summary of our Shopping Queries Data Set is given in the two tables below showing the statistics of the reduced and larger version, respectively. These tables include the number of unique queries, the number of judgements, and the average number of judgements per query (i.e., average depth) across the three different languages.

|       | Total | Total | Total | Train | Train | Train | Test | Test | Test |
| ------------- | ---------- | ------------- | ---------- | ---------- | ------------- | ---------- | ---------- | ------------- | ---------- |
| Language      | \# Queries | \# Judgements | Avg. Depth | \# Queries | \# Judgements | Avg. Depth | \# Queries | \# Judgements | Avg. Depth |
| English (US)  | 29,844     | 601,354       | 20.15      | 20,888     | 419,653       | 20.09      | 8,956      | 181,701       | 20.29      |
| Spanish (ES)  | 8,049      | 218,774       | 27.18      | 5,632      | 152,891       | 27.15      | 2,417      | 65,883        | 27.26      |
| Japanese (JP) | 10,407     | 297,883       | 28.62      | 7,284      | 209,094       | 28.71      | 3,123      | 88,789        | 28.43      |
| Overall       | 48,300     | 1,118,011     | 23.15      | 33,804     | 781,638       | 23.12      | 14,496     | 336,373       | 23.20      |

***Table 1**: Summary of the Shopping queries data set for task 1 (reduced version) - the number of unique queries, the number of judgements, and the average number of judgements per query.*

|       | Total | Total | Total | Train | Train | Train | Test | Test | Test |
| ------------- | ---------- | ------------- | ---------- | ---------- | ------------- | ---------- | ---------- | ------------- | ---------- |
| Language      | \# Queries | \# Judgements | Avg. Depth | \# Queries | \# Judgements | Avg. Depth | \# Queries | \# Judgements | Avg. Depth |
| English (US)  | 97,345     | 1,818,825     | 18.68      | 74,888     | 1,393,063     | 18.60      | 22,458     | 425,762       | 18.96      |
| Spanish (ES)  | 15,180     | 356,410       | 23.48      | 11,336     | 263,063       | 23.21      | 3,844      | 93,347        | 24.28      |
| Japanese (JP) | 18,127     | 446,053       | 24.61      | 13,460     | 327,146       | 24.31      | 4,667      | 118,907       | 25.48      |
| Overall       | 130,652    | 2,621,288     | 20.06      | 99,684     | 1,983,272     | 19.90      | 30,969     | 638,016       | 20.60      |

***Table 2**: Summary of the Shopping queries data set for tasks 2 and 3 (larger version) - the number of unique queries, the number of judgements, and the average number of judgements per query.*

## Usage

The [dataset](https://github.com/amazon-research/esci-code/tree/main/shopping_queries_dataset) has the following files:
- `shopping_queries_dataset_examples.parquet` contains the following columns : `example_id`, `query`, `query_id`, `product_id`, `product_locale`, `esci_label`, `small_version`, `large_version`, `split`
- `shopping_queries_dataset_products.parquet` contains the following columns : `product_id`, `product_title`, `product_description`, `product_bullet_point`, `product_brand`, `product_color`, `product_locale`
- `shopping_queries_dataset_sources.csv` contains the following columns : `query_id`, `source`

### Load examples, products and sources

```
import pandas as pd
df_examples = pd.read_parquet('shopping_queries_dataset_examples.parquet')
df_products = pd.read_parquet('shopping_queries_dataset_products.parquet')
df_sources = pd.read_csv("shopping_queries_dataset_sources.csv")
```

### Merge examples with products
```
df_examples_products = pd.merge(
    df_examples,
    df_products,
    how='left',
    left_on=['product_locale','product_id'],
    right_on=['product_locale', 'product_id']
)
```
### Filter and prepare for Task 1

```
df_task_1 = df_examples_products[df_examples_products["small_version"] == 1]
df_task_1_train = df_task_1[df_task_1["split"] == "train"]
df_task_1_test = df_task_1[df_task_1["split"] == "test"]
```

### Filter and prepare data for Task 2
```
df_task_2 = df_examples_products[df_examples_products["large_version"] == 1]
df_task_2_train = df_task_2[df_task_2["split"] == "train"]
df_task_2_test = df_task_2[df_task_2["split"] == "test"]
```

### Filter and prepare data for Task 3
```
df_task_3 = df_examples_products[df_examples_products["large_version"] == 1]
df_task_3["subtitute_label"] = df_task_3["esci_label"].apply(lambda esci_label: 1 if esci_label == "S" else 0 )
del df_task_3["esci_label"]
df_task_3_train = df_task_3[df_task_3["split"] == "train"]
df_task_3_test = df_task_3[df_task_3["split"] == "test"]
```
    
### Merge queries with sources (optional)
```
df_examples_products_source = pd.merge(
    df_examples_products,
    df_sources,
    how='left',
    left_on=['query_id'],
    right_on=['query_id']
)
```

## Baselines
In order to ensure the feasibility of the proposed tasks, we will provide the results obtained by standard baseline models run on this data sets. For example, for the first task (ranking), we have run a BERT model. For the remaining two tasks (classification) we will provide the results of the multilingual BERT-based models as the initial baseline.


### Requirements
We launched the baselines experiments creating an environment with Python 3.6 and installing the packages dependencies shown below:
```
numpy==1.19.2
pandas==1.1.5
transformers==4.16.2
scikit-learn==0.24.1
sentence-transformers==2.1.0
```

For installing the dependencies we can launch the following command:
```bash
pip install -r requirements.txt
```

### Reproduce published results

For a task **K**, we provide the same scripts, one for training the model (and preprocessing the data for tasks 2 and 3): `launch-experiments-taskK.sh`; and a second script for getting the predictions for the public test set using the model trained on the previous step: `launch-predictions-taskK.sh`.

#### Task 1 - Query Product Ranking

For task 1, we fine-tuned 3 models one for each `product_locale`.

For `us` locacale we fine-tuned [MS MARCO Cross-Encoders](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2). For `es` and `jp` locales [multilingual MPNet](https://huggingface.co/sentence-transformers/all-mpnet-base-v1). We used the query and title of the product as input for these models.

To get the nDCG score of the ranking models is needed the `terrier` source code (download version 5.5 [here](http://terrier.org/download/))

```bash
cd ranking/
./launch-experiments-task1.sh
./launch-predictions-task1.sh $TERRIER_PATH
```

#### Task 2 - Multiclass Product Classification

For task 2, we trained a Multilayer perceptron (MLP) classifier whose input is the concatenation of the representations provided by [BERT multilingual base](https://huggingface.co/bert-base-multilingual-uncased) for the query and title of the product.

```bash
cd classification_identification/
./launch-experiments-task2.sh
./launch-predictions-task2.sh
```

#### Task 3 - Product Substitute Identification

For task 3, we followed the same approach as in task 2.

```bash
cd classification_identification/
./launch-experiments-task3.sh
./launch-predictions-task3.sh
```

### Results
The following table shows the baseline results obtained through the different public tests of the three tasks.

| Task |  Metrics  | Scores |
|:----:|:--------:|:-----:|
|    1 | nDCG     | 0.83 |
|    2 | Macro F1, Micro F1 | 0.23, 0.62 |
|    3 | Macro F1, Micro F1 | 0.44, 0.76 |

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
