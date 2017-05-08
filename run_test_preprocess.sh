##!/usr/bin/env bash
#python ./run.py \
#-d ./data/preprocessed/ml-1m/0.2/ \
#-a ./data/preprocessed/ml-1m/ \
#-c True \
#-r ./data/rare/movielens/ml-1m_ratings.dat \
#-i ./data/rare/movielens/ml_plot.dat \
#-m 1


#!/usr/bin/env bash
python ./run.py \
-d ./data/preprocessed/aiv/0.2/ \
-a ./data/preprocessed/aiv/ \
-c True \
-r ./data/rare/aiv/Amazon_Instant_Video_ratings.txt \
-i ./data/rare/aiv/Amazon_Instant_Video_items.txt \
-m 1