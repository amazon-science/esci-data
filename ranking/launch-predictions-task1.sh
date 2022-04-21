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

DATA_TASK1_PATH="../data/task1/"
PRODUCT_CATALOGUE_PATH_FILE="${DATA_TASK1_PATH}/product_catalogue-v0.2.csv.zip"
TEST_PATH_FILE="${DATA_TASK1_PATH}/test_public-v0.2.csv.zip"
MODELS_TASK1_PATH="./models"
HYPOTHESIS_TASK1_PATH="./hypothesis"
mkdir -p ${HYPOTHESIS_TASK1_PATH}

for i in "${!LOCALES[@]}"
do
    N_DEV_QUERIES="${N_DEV_QUERIES_LOCALE[$i]}"
    LOCALE="${LOCALES[$i]}"
    MODEL_PATH="${MODELS_TASK1_PATH}/task_1_ranking_model_${LOCALE}"
    HYPOTHESIS_PATH_FILE="${HYPOTHESIS_TASK1_PATH}/task_1_ranking_model_${LOCALE}.csv"
    python inference.py \
        ${TEST_PATH_FILE} \
        ${PRODUCT_CATALOGUE_PATH_FILE} \
        ${LOCALE} \
        ${MODEL_PATH} \
        ${HYPOTHESIS_PATH_FILE}
done

# Combine predictions from three locales into one file: us + es + jp
python concat_predictions.py ${HYPOTHESIS_TASK1_PATH}