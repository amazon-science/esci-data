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
import pandas as pd
import numpy as np
import os


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Directory where the dataset is stored.")
    parser.add_argument("hypothesis_folder_path", type=str, help="Directory where the hypothesis are stored.")
    parser.add_argument("--output_path", type=str, default="./output", help="Directory where the generated files are stored.")
    args = parser.parse_args()

    """ 0. Init variables """

    col_iteration = "iteration"
    col_query_id = "query_id"
    col_product_id = "product_id" 
    col_product_locale = "product_locale"
    col_small_version = "small_version"
    col_split = "split"
    col_esci_label = "esci_label"
    col_relevance_pos = "relevance_pos"
    col_ranking_postion = "ranking_postion"
    col_score = "score"
    col_conf = "conf"

    max_trec_eval_score = 128
    min_trec_eval_score = 0

    esci_label2relevance_pos = {
        "E" : 4,
        "S" : 2,
        "C" : 3,
        "I" : 1,
    }

    """ 1. Generate RESULTS file """

    locales = [
        "us",
        "es",
        "jp",
    ]

    df_results = pd.DataFrame()
    for locale in locales:
        df_ = pd.read_csv(
            os.path.join(args.hypothesis_folder_path, f"task_1_ranking_model_{locale}.csv"),
        )
        df_results = pd.concat([df_results, df_])
    
    df_results_product_id = df_results.groupby(by=[col_query_id])
    l_query_id = []
    l_product_id = []
    l_ranking_postion = []
    l_score = []
    for (query_id, rows) in df_results_product_id:
        n = len(rows)
        l_query_id += [query_id for _ in range(n)]
        l_product_id += rows[col_product_id].to_list()
        l_ranking_postion += [i for i in range(n)]
        l_score += list(np.arange(min_trec_eval_score, max_trec_eval_score, max_trec_eval_score / n).round(3)[::-1][:n])
        
    df_res = pd.DataFrame({
        col_query_id : l_query_id,
        col_product_id : l_product_id,
        col_ranking_postion : l_ranking_postion,
        col_score : l_score,
    })
    model_name_value = "baseline"
    iteration_value = "Q0"
    df_res[col_conf] = model_name_value
    df_res[col_iteration] = iteration_value

    df_res[[
        col_query_id,
        col_iteration,
        col_product_id,
        col_ranking_postion,
        col_score,
        col_conf,
    ]].to_csv(
        os.path.join(args.output_path, "hypothesis.results"),
        index=False,
        header=False,
        sep=' ',
    )


    """ 2. Generate QRELS file """
    df_examples = pd.read_parquet(os.path.join(args.dataset_path, 'shopping_queries_dataset_examples.parquet'))
    df_products = pd.read_parquet(os.path.join(args.dataset_path, 'shopping_queries_dataset_products.parquet'))
    df_examples_products = pd.merge(
        df_examples,
        df_products,
        how='left',
        left_on=[col_product_locale, col_product_id],
        right_on=[col_product_locale, col_product_id]
    )
    df_examples_products = df_examples_products[df_examples_products[col_small_version] == 1]
    df_examples_products = df_examples_products[df_examples_products[col_split] == "test"]

    df_examples_products[col_iteration] = 0
    df_examples_products[col_relevance_pos] = df_examples_products[col_esci_label].apply(lambda esci_label: esci_label2relevance_pos[esci_label])
    df_examples_products = df_examples_products[[
            col_query_id,
            col_iteration,
            col_product_id,
            col_relevance_pos,
    ]]
    df_examples_products.to_csv(
        os.path.join(args.output_path, "test.qrels"),
        index=False,
        header=False,
        sep=' ',
    )

    #../code/terrier-project-5.5/bin/terrier trec_eval ${QRELS_FILE} ${RES_FILE} -c -J -m 'ndcg.1=0,2=0.01,3=0.1,4=1'

if __name__ == "__main__": 
    main()