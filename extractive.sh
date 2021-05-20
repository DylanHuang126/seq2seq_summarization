#!/usr/bin/env bash

python3.6 ./src/preprocess_seq_tag_test.py --test_data_path $1
python3.6 ./src/eval_ext.py --output_path $2

