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

RANDOM_STATE=42
BATCH_SIZE=128
BERT_MODEL_NAME="bert-base-multilingual-uncased"
BERT_MAX_LENGTH=256
BERT_SIZE=768
LABELS_TYPE="substitute_identification"

SQD_PATH="../shopping_queries_dataset/"
DATA_REPRESENTATIONS_PATH="./text_representations/task3"
DICT_PRODUCTS_PATH_FILE="${DATA_REPRESENTATIONS_PATH}/dict_products_train.npy"
DICT_QUERIES_PATH_FILE="${DATA_REPRESENTATIONS_PATH}/dict_examples_train.npy"

ARRAY_PRODUCTS_PATH_FILE="${DATA_REPRESENTATIONS_PATH}/array_products_train.npy"
ARRAY_QUERIES_PATH_FILE="${DATA_REPRESENTATIONS_PATH}/array_queries_train.npy"
ARRAY_LABELS_PATH_FILE="${DATA_REPRESENTATIONS_PATH}/array_labels_train.npy"

# 1. Get BERT representations for queries and products
mkdir -p ${DATA_REPRESENTATIONS_PATH}
python compute_bert_representations.py \
    ${SQD_PATH} \
    "train" \
    --output_queries_path_file ${DICT_QUERIES_PATH_FILE} \
    --output_product_catalogue_path_file ${DICT_PRODUCTS_PATH_FILE} \
    --model_name ${BERT_MODEL_NAME} \
    --bert_max_length ${BERT_MAX_LENGTH} \
    --batch_size ${BATCH_SIZE}

# 2. Build inputs dataset from BERT representations
python build_input_data_model.py \
    ${SQD_PATH} \
    "train" \
    ${DICT_PRODUCTS_PATH_FILE} \
    ${DICT_QUERIES_PATH_FILE} \
    ${ARRAY_QUERIES_PATH_FILE} \
    ${ARRAY_PRODUCTS_PATH_FILE} \
    ${ARRAY_LABELS_PATH_FILE} \
    --labels_type ${LABELS_TYPE} \
    --bert_size ${BERT_SIZE}

MODELS_PATH="./models"
MODEL_SAVE_PATH="${MODELS_PATH}/task_3_substitute_identification_model"
RANDOM_STATE=42
BATCH_SIZE=256
WEIGHT_DECAY=0.01
NUM_TRAIN_EPOCHS=4
LR=5e-5
EPS=1e-8
NUM_WARMUP_STEPS=0
MAX_GRAD_NORM=1
VALIDATION_STEPS=250
NUM_DEV_EXAMPLES=5505

# 3. Perform the training
python train.py \
    ${ARRAY_QUERIES_PATH_FILE} \
    ${ARRAY_PRODUCTS_PATH_FILE} \
    ${ARRAY_LABELS_PATH_FILE} \
    ${MODEL_SAVE_PATH} \
    ${LABELS_TYPE} \
    --random_state ${RANDOM_STATE} \
    --batch_size ${BATCH_SIZE} \
    --weight_decay ${WEIGHT_DECAY} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --lr ${LR} \
    --eps ${EPS} \
    --num_warmup_steps ${NUM_WARMUP_STEPS} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --validation_steps ${VALIDATION_STEPS} \
    --num_dev_examples ${NUM_DEV_EXAMPLES}