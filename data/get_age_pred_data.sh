#!/usr/bin/env bash

mkdir data_age_pred
cd data_age_pred

curl -o transactions_train.csv.gz -L 'https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/transactions_train.csv.gz?download=true'
curl -o transactions_test.csv.gz -L 'https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/transactions_test.csv.gz?download=true'
curl -o train_target.csv -L 'https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/train_target.csv?download=true'

gunzip -f *.csv.gz
cd ..

mkdir -p data_age_pred/results

python -m ptls.make_datasets \
    --data_path data_age_pred/ \
    --trx_files transactions_train.csv transactions_test.csv \
    --col_client_id "client_id" \
    --cols_event_time "#float" "trans_date" \
    --cols_category "trans_date" "small_group" \
    --cols_log_norm "amount_rur" \
    --target_files train_target.csv \
    --col_target bins \
    --test_size 0.1 \
    --output_train_path "data_age_pred/train_trx.p" \
    --output_test_path "data_age_pred/test_trx.p" \
    --output_test_ids_path "data_age_pred/test_ids.csv" \
    --log_file "data_age_pred/results/dataset_age_pred.txt"
