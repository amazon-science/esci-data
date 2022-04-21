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
import os


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("hypothesis_folder_path", type=str, help="Directory where the hypothesis are stored.")
    args = parser.parse_args()

    locales = [
        "us",
        "es",
        "jp",
    ]

    df = pd.DataFrame()
    for locale in locales:
        df_ = pd.read_csv(
            os.path.join(args.hypothesis_folder_path, f"task_1_ranking_model_{locale}.csv"),
        )
        df = pd.concat([df, df_])
    
    df.to_csv(
        os.path.join(args.hypothesis_folder_path, f"task_1_ranking_model.csv"),
        index=False,
        sep=',',
    )


if __name__ == "__main__": 
    main()