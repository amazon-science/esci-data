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

if [ "$#" -ne 1 ]; then
    echo './launch-predictions-task1.sh ${BIN_TERRIER_CORE_PATH}'
    printf "Example: ./launch-predictions-task1.sh /tmp/terrier-project-5.5/bin/terrier\n"
    printf "Read https://github.com/terrier-org/terrier-core for more details\n"
    exit
fi

LOCALES=("us")
LOCALES+=("es")
LOCALES+=("jp")

DATA_SQD="../shopping_queries_dataset/"
MODELS_TASK1_PATH="./models"
HYPOTHESIS_TASK1_PATH="./hypothesis"
TREC_EVAL_DATA_PATH="./trec_eval_data"

mkdir -p ${HYPOTHESIS_TASK1_PATH}

for i in "${!LOCALES[@]}"
do
    N_DEV_QUERIES="${N_DEV_QUERIES_LOCALE[$i]}"
    LOCALE="${LOCALES[$i]}"
    MODEL_PATH="${MODELS_TASK1_PATH}/task_1_ranking_model_${LOCALE}"
    HYPOTHESIS_PATH_FILE="${HYPOTHESIS_TASK1_PATH}/task_1_ranking_model_${LOCALE}.csv"
    python inference.py \
        ${DATA_SQD} \
        ${LOCALE} \
        ${MODEL_PATH} \
        ${HYPOTHESIS_PATH_FILE}
done

# Compute nDCG score (terrier trec_eval path is needed)[see https://github.com/terrier-org/terrier-core]
mkdir -p ${TREC_EVAL_DATA_PATH}
python prepare_trec_eval_files.py ${DATA_SQD} \
    ${HYPOTHESIS_TASK1_PATH} \
    --output_path ${TREC_EVAL_DATA_PATH}

$1/terrier trec_eval "${TREC_EVAL_DATA_PATH}/test.qrels" "${TREC_EVAL_DATA_PATH}/hypothesis.results" -c -J -m 'ndcg.1=0,2=0.01,3=0.1,4=1'