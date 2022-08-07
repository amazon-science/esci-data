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

LOCALES=("us")
LOCALES+=("es")
LOCALES+=("jp")

N_DEV_QUERIES_LOCALE=(400)
N_DEV_QUERIES_LOCALE+=(200)
N_DEV_QUERIES_LOCALE+=(200)

RANDOM_STATE=42
TRAIN_BATCH_SIZE=32

DATA_SQD="../shopping_queries_dataset/"
MODELS_TASK1_PATH="./models"

for i in "${!LOCALES[@]}"
do
    N_DEV_QUERIES="${N_DEV_QUERIES_LOCALE[$i]}"
    LOCALE="${LOCALES[$i]}"
    MODEL_SAVE_PATH="${MODELS_TASK1_PATH}/task_1_ranking_model_${LOCALE}"
    python train.py \
        ${DATA_SQD} \
        ${LOCALE} \
        ${MODEL_SAVE_PATH} \
        --random_state ${RANDOM_STATE} \
        --n_dev_queries ${N_DEV_QUERIES} \
        --train_batch_size ${TRAIN_BATCH_SIZE}
done