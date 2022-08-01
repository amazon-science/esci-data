# The Shopping queries dataset
## Description
The dataset has the following files:
- `shopping_queries_dataset_examples.parquet` contains the following columns : `example_id`, `query`, `query_id`, `product_id`, `product_locale`, `esci_label`, `small_version`, `large_version`, `split`
- `shopping_queries_dataset_products.parquet` contains the following columns : `product_id`, `product_title`, `product_description`, `product_bullet_point`, `product_brand`, `product_color`, `product_locale`
- `shopping_queries_dataset_sources.csv` contains the following columns : `query_id`, `source`

## Statistics
### Task 1 (small version)

|       | Total | Total | Total | Train | Train | Train | Test | Test | Test |
| ------------- | ---------- | ------------- | ---------- | ---------- | ------------- | ---------- | ---------- | ------------- | ---------- |
| Language      | \# Queries | \# Judgements | Avg. Depth | \# Queries | \# Judgements | Avg. Depth | \# Queries | \# Judgements | Avg. Depth |
| English (US)  | 29,844     | 601,354       | 20.15      | 20,888     | 419,653       | 20.09      | 8,956      | 181,701       | 20.29      |
| Spanish (ES)  | 8,049      | 218,774       | 27.18      | 5,632      | 152,891       | 27.15      | 2,417      | 65,883        | 27.26      |
| Japanese (JP) | 10,407     | 297,883       | 28.62      | 7,284      | 209,094       | 28.71      | 3,123      | 88,789        | 28.43      |
| Overall       | 48,300     | 1,118,011     | 23.15      | 33,804     | 781,638       | 23.12      | 14,496     | 336,373       | 23.20      |


### Tasks 2 and 3 (large version)
|       | Total | Total | Total | Train | Train | Train | Test | Test | Test |
| ------------- | ---------- | ------------- | ---------- | ---------- | ------------- | ---------- | ---------- | ------------- | ---------- |
| Language      | \# Queries | \# Judgements | Avg. Depth | \# Queries | \# Judgements | Avg. Depth | \# Queries | \# Judgements | Avg. Depth |
| English (US)  | 97,345     | 1,818,825     | 18.68      | 74,888     | 1,393,063     | 18.60      | 22,458     | 425,762       | 18.96      |
| Spanish (ES)  | 15,180     | 356,410       | 23.48      | 11,336     | 263,063       | 23.21      | 3,844      | 93,347        | 24.28      |
| Japanese (JP) | 18,127     | 446,053       | 24.61      | 13,460     | 327,146       | 24.31      | 4,667      | 118,907       | 25.48      |
| Overall       | 130,652    | 2,621,288     | 20.06      | 99,684     | 1,983,272     | 19.90      | 30,969     | 638,016       | 20.60      |

### Dsitribution
|  | Total      | Total      | Total     | Total      | Train      | Train      | Train     | Train      | Test      | Test      | Test     | Test      |
| --------------- | ------ | ------ | ----- | ------ | ------ | ------ | ----- | ------ | ------ | ------ | ----- | ------ |
| Dataset version | E      | S      | C     | I      | E      | S      | C     | I      | E      | S      | C     | I      |
| Small           | 43.78% | 34.30% | 5.15% | 16.76% | 43.64% | 34.28% | 5.19% | 16.89% | 44.11% | 34.36% | 5.06% | 16.47% |
| Large           | 65.16% | 21.91% | 2.89% | 10.04% | 66.29% | 21.23% | 2.76% | 9.72%  | 61.65% | 24.04% | 3.27% | 11.04% |


## Usage

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