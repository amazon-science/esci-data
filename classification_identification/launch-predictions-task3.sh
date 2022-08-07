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
LABELS_TYPE="substitute_identification"

SQD_PATH="../shopping_queries_dataset/"
DATA_REPRESENTATIONS_PATH="./text_representations/task3"

DICT_PRODUCTS_PATH_FILE="${DATA_REPRESENTATIONS_PATH}/dict_products_test.npy"
DICT_QUERIES_PATH_FILE="${DATA_REPRESENTATIONS_PATH}/dict_examples_test.npy"

ARRAY_PRODUCTS_PATH_FILE="${DATA_REPRESENTATIONS_PATH}/array_product_test.npy"
ARRAY_QUERIES_PATH_FILE="${DATA_REPRESENTATIONS_PATH}/array_queries_test.npy"
ARRAY_LABELS_PATH_FILE="${DATA_REPRESENTATIONS_PATH}/array_labels_test.npy"

mkdir -p ${DATA_REPRESENTATIONS_PATH}

# 1. Get BERT representations for queries and products
python compute_bert_representations.py \
    ${SQD_PATH} \
    "test" \
    --output_queries_path_file ${DICT_QUERIES_PATH_FILE} \
    --output_product_catalogue_path_file ${DICT_PRODUCTS_PATH_FILE} \
    --model_name ${BERT_MODEL_NAME} \
    --bert_max_length ${BERT_MAX_LENGTH} \
    --batch_size ${BATCH_SIZE}

# 2. Build inputs datasets from BERT representations
python build_input_data_model.py \
    ${SQD_PATH} \
    "test" \
    ${DICT_PRODUCTS_PATH_FILE} \
    ${DICT_QUERIES_PATH_FILE} \
    ${ARRAY_QUERIES_PATH_FILE} \
    ${ARRAY_PRODUCTS_PATH_FILE} \
    ${ARRAY_LABELS_PATH_FILE} \
    --bert_size ${BERT_SIZE} \
    --labels_type ${LABELS_TYPE} 

MODELS_PATH="./models"
MODEL_PATH="${MODELS_PATH}/task_3_substitute_identification_model"
BATCH_SIZE=256
SUBSTITUTE_IDENTIFICATION_TH=0.5
HYPOTHESIS_PATH="./hypothesis"
HYPOTHESIS_PATH_FILE="${HYPOTHESIS_PATH}/task_3_substitute_identification_model.csv"
mkdir -p ${HYPOTHESIS_PATH}

# 3. Perform the predictions
python inference.py \
    ${SQD_PATH} \
    "test" \
    ${ARRAY_QUERIES_PATH_FILE} \
    ${ARRAY_PRODUCTS_PATH_FILE} \
    ${TEST_PUBLIC_PATH_FILE} \
    ${MODEL_PATH} \
    ${LABELS_TYPE} \
    ${HYPOTHESIS_PATH_FILE} \
    --batch_size ${BATCH_SIZE} \
    --substitute_identification_th ${SUBSTITUTE_IDENTIFICATION_TH}