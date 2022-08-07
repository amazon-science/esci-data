# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# 
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
#  
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Directory where the dataset is stored.")
    parser.add_argument("split", type=str, choices=['train', 'test'], help="Split of the dataset.")
    parser.add_argument("dict_products_path_file", type=str, help="Input file with the mapping of the products to BERT representations.")
    parser.add_argument("dict_queries_path_file", type=str, help="Input file with the mapping of the queries to BERT representations.")
    parser.add_argument("output_queries_path_file", type=str, help="Output array file with the BERT representations of the queries.")
    parser.add_argument("output_products_path_file", type=str, help="Output array file with the BERT representations of the products.")
    parser.add_argument("output_labels_path_file", type=str, default=None, help="Output array file the labels.")
    parser.add_argument("--labels_type", type=str, choices=["esci_labels", "substitute_identification"], help="Task: esci_labels | substitute_identification.")
    parser.add_argument("--bert_size", type=int, default=768, help="BERT embeddings dimension.")
    args = parser.parse_args()

    """" 1. Init variables """
    col_label = "esci_label"
    col_query_id = "query_id"
    col_product_id = "product_id" 
    col_large_version = "large_version"
    col_split = "split"
    col_esci_label = "esci_label"

    dict_labels_type = dict()
    dict_labels_type['esci_labels'] = {
        'E' : 0,
        'S' : 1,
        'C' : 2,
        'I' : 3,
    }
    dict_labels_type['substitute_identification'] = {
        'no_substitute' : 0,
        'substitute' : 1,
    }

    """" 2. Load data """
    dict_products = np.load(args.dict_products_path_file, allow_pickle=True)
    dict_queries = np.load(args.dict_queries_path_file, allow_pickle=True)
    df = pd.read_parquet(os.path.join(args.dataset_path, 'shopping_queries_dataset_examples.parquet'))
    df = df[df[col_large_version] == 1]
    df = df[df[col_split] == args.split]
    
    df = df[[
        col_query_id,
        col_product_id,
        col_esci_label,
    ]]
    
    if args.labels_type == "substitute_identification":
        tmp_dict = {
            'E' : 'no_substitute',
            'S' : 'substitute',
            'C' : 'no_substitute',
            'I' : 'no_substitute',
        }
        df[col_esci_label] = df[col_esci_label].apply(lambda esci_label: tmp_dict[esci_label])

    num_examples = len(df)
    array_queries = np.zeros((num_examples, args.bert_size))
    array_products = np.zeros((num_examples, args.bert_size))
    
    """" 3. Map queries and products """
    for i in tqdm(range(num_examples)):
        array_queries[i] = dict_queries[()][df.iloc[i][col_query_id]]
        array_products[i] = dict_products[()][df.iloc[i][col_product_id]]
        
    """" 4. Export representations for queries and products """
    np.save(args.output_queries_path_file, array_queries)
    np.save(args.output_products_path_file, array_products)

    """" 5. Export labels (only for training) """
    col_class_id = 'class_id'
    labels2class_id = dict_labels_type[args.labels_type]
    df[col_class_id] = df[col_label].apply(lambda label: labels2class_id[label])
    np.save(args.output_labels_path_file, df[col_class_id].to_numpy())


if __name__ == "__main__": 
    main()  