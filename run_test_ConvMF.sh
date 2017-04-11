#!/usr/bin/env bash
python ./run.py \
-d ./data/preprocessed/ml-1m/0.2/ \
-a ./data/preprocessed/ml-1m/ \
-o ./test/ml-1m/result/1_100_200 \
-e 200 \
-p ./data/preprocessed/glove/glove.6B.200d.txt \
-u 10 \
-v 100 \
-g True
