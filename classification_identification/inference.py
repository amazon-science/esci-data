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
import torch
from sklearn.metrics import f1_score
from query_product import QueryProductClassifier, generate_dataset
import os
from tqdm import tqdm
import pandas as pd


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Directory where the dataset is stored.")
    parser.add_argument("split", type=str, choices=['train', 'test'], help="Split of the dataset.")
    parser.add_argument("test_queries_path_file", type=str, help="Input array file with the BERT representations of the queries.")
    parser.add_argument("test_products_path_file", type=str, help="Input array file with the BERT representations of the products.")
    parser.add_argument("model_path", type=str, help="Directory where the model is stored.")
    parser.add_argument("task", type=str, choices=["esci_labels", "substitute_identification"], help="Task: esci_labels | substitute_identification.")
    parser.add_argument("hypothesis_path_file", type=str, help="Output CSV with the hypothesis.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--substitute_identification_th", type=float, default=0.5, help="Binary classification threshold for the substitute_identification task.")
    args = parser.parse_args()

    """ 0. Init variables """
    class_id2esci_label = {
        0 : 'E',
        1 : 'S',
        2 : 'C',
        3 : 'I',
    }
    class_id2substitute_identification = {
        0 : 'no_substitute',
        1 : 'substitute',
    }
    task2num_labels = {
        "esci_labels" : (4, class_id2esci_label, "esci_label"),
        "substitute_identification" : (2, class_id2substitute_identification, "substitute_label"),
    }
    col_example_id = "example_id"
    col_gold_label = "gold_label"
    col_large_version = "large_version"
    col_split = "split"
    col_esci_label = "esci_label"

    """ 1. Load data """
    query_array = np.load(args.test_queries_path_file)
    asin_array = np.load(args.test_products_path_file)
    labels_array = np.zeros(asin_array.shape[0])
    num_labels, class_id2label, col_label = task2num_labels[args.task]
    dataset = generate_dataset(
        query_array,
        asin_array,
        labels_array,
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    """ 2. Prepare model """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QueryProductClassifier(num_labels=num_labels)
    model.load_state_dict(torch.load(
        os.path.join(args.model_path, "pytorch_model.bin"),
    ))
    model.to(device)

    """ 3. Generate hypothesis"""
    a_hypothesis = np.array([])
    model.eval()
    with torch.no_grad():
        for test_batch in tqdm(data_loader):
            # 0: query_encoding, 1: product_encoding
            with torch.no_grad():
                logits = model(test_batch[0].to(device), test_batch[1].to(device))
                if args.task == "esci_labels":
                    logits = logits.detach().cpu().numpy()
                    hypothesis = np.argmax(logits, axis=1)
                else:
                    output = torch.sigmoid(logits)
                    output = output.type(torch.FloatTensor)
                    output = output.detach().cpu().numpy()
                    hypothesis = np.digitize(output, [args.substitute_identification_th])
                a_hypothesis = np.concatenate([
                    a_hypothesis,
                    hypothesis,
                ])
    a_hypothesis = a_hypothesis.astype(int)

    """ 4. Prepare hypothesis file """
    df = pd.read_parquet(os.path.join(args.dataset_path, 'shopping_queries_dataset_examples.parquet'))
    df = df[df[col_large_version] == 1]
    df = df[df[col_split] == args.split]
    labels = [ class_id2label[int(hyp)] for hyp in a_hypothesis ]
    if args.task == "substitute_identification":
        tmp_dict = {
            'E' : 'no_substitute',
            'S' : 'substitute',
            'C' : 'no_substitute',
            'I' : 'no_substitute',
        }
        df[col_esci_label] = df[col_esci_label].apply(lambda esci_label: tmp_dict[esci_label])
    df_hypothesis = pd.DataFrame({
        col_example_id : df[col_example_id].to_list(),
        col_label : labels,
        col_gold_label : df[col_esci_label].to_list()
    })
    df_hypothesis[[col_example_id, col_label]].to_csv(
        args.hypothesis_path_file,
        index=False,
        sep=',',
    )
    macro_f1 = f1_score(
        df_hypothesis[col_gold_label], 
        df_hypothesis[col_label], 
        average='macro',
    )
    micro_f1 = f1_score(
        df_hypothesis[col_gold_label], 
        df_hypothesis[col_label], 
        average='micro',
        )
    print("macro\tmicro")
    print(f"{macro_f1:.3f}\t{micro_f1:.3f}")


if __name__ == "__main__": 
    main()  
