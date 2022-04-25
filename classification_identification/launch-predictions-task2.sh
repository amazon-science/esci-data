#!/bin/bash
#
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

BATCH_SIZE=128
BERT_MODEL_NAME="bert-base-multilingual-uncased"
BERT_MAX_LENGTH=256
BERT_SIZE=768
LABELS_TYPE="esci_labels"

DATA_TASK2_PATH="../data/task2"
TEST_PUBLIC_PATH_FILE="${DATA_TASK2_PATH}/test_public-v0.2.csv.zip"

DATA_REPRESENTATIONS_PATH="./text_representations/task2"
DICT_PRODUCTS_PATH_FILE="${DATA_REPRESENTATIONS_PATH}/dict_product_catalogue-v0.2.npy"
DICT_QUERIES_PATH_FILE="${DATA_REPRESENTATIONS_PATH}/dict_test_public-v0.2.npy"

ARRAY_PRODUCTS_PATH_FILE="${DATA_REPRESENTATIONS_PATH}/array_product_test_public-v0.2.npy"
ARRAY_QUERIES_PATH_FILE="${DATA_REPRESENTATIONS_PATH}/array_queries_test_public-v0.2.npy"

mkdir -p ${DATA_REPRESENTATIONS_PATH}

# 1. Get BERT representations for queries and products
python compute_bert_representations.py \
    --input_queries_path_file ${TEST_PUBLIC_PATH_FILE} \
    --output_queries_path_file ${DICT_QUERIES_PATH_FILE} \
    --model_name ${BERT_MODEL_NAME} \
    --bert_max_length ${BERT_MAX_LENGTH} \
    --batch_size ${BATCH_SIZE}

# 2. Build inputs datasets from BERT representations
python build_input_data_model.py \
    ${DICT_PRODUCTS_PATH_FILE} \
    ${DICT_QUERIES_PATH_FILE} \
    ${TEST_PUBLIC_PATH_FILE} \
    ${TRAIN_PATH_FILE} \
    ${ARRAY_QUERIES_PATH_FILE} \
    ${ARRAY_PRODUCTS_PATH_FILE} \
    --bert_size ${BERT_SIZE}

MODELS_PATH="./models"
MODEL_PATH="${MODELS_PATH}/task_2_esci_classifier_model"
BATCH_SIZE=256
HYPOTHESIS_PATH="./hypothesis"
HYPOTHESIS_PATH_FILE="${HYPOTHESIS_PATH}/task_2_esci_classifier_model.csv"
mkdir -p ${HYPOTHESIS_PATH}

# 3. Perform the predictions
python inference.py \
    ${ARRAY_QUERIES_PATH_FILE} \
    ${ARRAY_PRODUCTS_PATH_FILE} \
    ${TEST_PUBLIC_PATH_FILE} \
    ${MODEL_PATH} \
    ${LABELS_TYPE} \
    ${HYPOTHESIS_PATH_FILE} \
    --batch_size ${BATCH_SIZE}