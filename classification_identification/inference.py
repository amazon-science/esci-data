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
from query_product import QueryProductClassifier, generate_dataset
import os
from tqdm import tqdm
import pandas as pd


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("test_queries_path_file", type=str, help="Input array file with the BERT representations of the queries.")
    parser.add_argument("test_products_path_file", type=str, help="Input array file with the BERT representations of the products.")
    parser.add_argument("test_path_file", type=str, help="Input CSV with the pairs of queries and products.")
    parser.add_argument("model_path", type=str, help="Directory where the model is stored.")
    parser.add_argument("task", type=str, choices=["esci_labels", "substitute_identification"], help="Task: esci_labels | substitute_identification.")
    parser.add_argument("hypothesis_path_file", type=str, help="Output CSV with the hypothesis.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--substitute_identification_th", type=float, default=0.5, help="Binary classification threshold for the substitute_identification task.")
    args = parser.parse_args()

    """ 0. Init variables """
    class_id2esci_label = {
        0 : 'irrelevant',
        1 : 'substitute',
        2 : 'exact',
        3 : 'complement',
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
    df = pd.read_csv(args.test_path_file)
    labels = [ class_id2label[int(hyp)] for hyp in a_hypothesis ]
    df_hypothesis = pd.DataFrame({
        col_example_id : df[col_example_id].to_list(),
        col_label : labels,
    })
    df_hypothesis[[col_example_id, col_label]].to_csv(
        args.hypothesis_path_file,
        index=False,
        sep=',',
    )

    
if __name__ == "__main__": 
    main()  