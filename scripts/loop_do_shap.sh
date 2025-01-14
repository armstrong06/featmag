#!/bin/bash

# is_p=1
# if [ is_p ]; then
#     train_file='/uufs/chpc.utah.edu/common/home/koper-group3/alysha/magnitudes/feature_splits/p.train.csv'
# else
#     train_file='/uufs/chpc.utah.edu/common/home/koper-group3/alysha/magnitudes/feature_splits/s.train.csv'
# fi 

train_file='/uufs/chpc.utah.edu/common/home/koper-group3/alysha/magnitudes/feature_splits/p.train.csv'
echo "Loading stations from ${train_file}"
stats=`awk 'BEGIN{FS=","} NR>1 {print $4}' ${train_file} | sort | uniq`

for stat in $stats; do
   echo $stat
   nohup_file="../logs/SHAP/out.${stat}.P"
   echo $nohup_file
   nohup python -u do_shap.py -s $stat --is_p &> $nohup_file & 
done
wait