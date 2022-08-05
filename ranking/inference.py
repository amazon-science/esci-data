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
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from tqdm import tqdm


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Directory where the dataset is stored.")
    parser.add_argument("locale", type=str, choices=['us', 'es', 'jp'], help="Locale of the queries.")
    parser.add_argument("model_path", type=str, help="Directory where the model is stored.")
    parser.add_argument("hypothesis_path_file", type=str, help="Output CSV with the hypothesis.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    args = parser.parse_args()

    """ 0. Init variables """
    col_query = "query"
    col_query_id = "query_id"
    col_product_id = "product_id" 
    col_product_title = "product_title"
    col_product_locale = "product_locale"
    col_small_version = "small_version"
    col_split = "split"
    col_scores = "scores"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ 1. Load data """    
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
    df_examples_products = df_examples_products[df_examples_products[col_product_locale] == args.locale]
    
    features_query = df_examples_products[col_query].to_list()
    features_product = df_examples_products[col_product_title].to_list()
    n_examples = len(features_query)
    scores = np.zeros(n_examples)

    if args.locale == "us":
        """ 2. Prepare Cross-encoder model """
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        """ 3. Generate hypothesis """
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, n_examples, args.batch_size)):
                j = min(i + args.batch_size, n_examples)
                features_query_ = features_query[i:j]
                features_product_ = features_product[i:j]
                features = tokenizer(features_query_, features_product_,  padding=True, truncation=True, return_tensors="pt").to(device)
                scores[i:j] = np.squeeze(model(**features).logits.cpu().detach().numpy())
                i = j
    else :
        """ 2. Prepare Sentence transformer model """
        model = AutoModel.from_pretrained(args.model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        # CLS Pooling - Take output from first token
        def cls_pooling(model_output):
            return model_output.last_hidden_state[:,0]
        # Encode text
        def encode(texts):
            # Tokenize sentences
            encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input, return_dict=True)
            # Perform pooling
            embeddings = cls_pooling(model_output)
            return embeddings
        model.eval()
        
        """ 3. Generate hypothesis """
        with torch.no_grad():
            for i in tqdm(range(0, n_examples, args.batch_size)):
                j = min(i + args.batch_size, n_examples)
                features_query_ = features_query[i:j]
                features_product_ = features_product[i:j]
                query_emb = encode(features_query_)
                product_emb = encode(features_product_)
                scores[i:j] = torch.diagonal(torch.mm(query_emb, product_emb.transpose(0, 1)).to('cpu'))
                i = j
    
    """ 4. Prepare hypothesis file """   
    df_hypothesis = pd.DataFrame({
        col_query_id : df_examples_products[col_query_id].to_list(),
        col_product_id : df_examples_products[col_product_id].to_list(),
        col_scores : scores,
    })
    df_hypothesis = df_hypothesis.sort_values(by=[col_query_id, col_scores], ascending=False)
    df_hypothesis[[col_query_id, col_product_id]].to_csv(
        args.hypothesis_path_file,
        index=False,
        sep=',',
    )


if __name__ == "__main__": 
    main()