#!/usr/bin/env bash
python ./run.py \
-d ./data/preprocessed/ml-1m/0.2/ \
-a ./data/preprocessed/ml-1m/ \
-c True \
-r ./data/movielens/ml-1m_ratings.dat \
-i ./data/movielens/ml_plot.dat \
-m 1
