python3 src/inference.py --config-dir=configs/compare_hidden_dims/32 --config-name=inference_32_test
python3 src/inference.py --config-dir=configs/compare_hidden_dims/32 --config-name=inference_32_train

python3 src/inference.py --config-dir=configs/compare_hidden_dims/64 --config-name=inference_64_test
python3 src/inference.py --config-dir=configs/compare_hidden_dims/64 --config-name=inference_64_train

python3 src/inference.py --config-dir=configs/compare_hidden_dims/128 --config-name=inference_128_test
python3 src/inference.py --config-dir=configs/compare_hidden_dims/128 --config-name=inference_128_train

python3 src/inference.py --config-dir=configs/compare_hidden_dims/256 --config-name=inference_256_test
python3 src/inference.py --config-dir=configs/compare_hidden_dims/256 --config-name=inference_256_train

python3 src/inference.py --config-dir=configs/compare_hidden_dims/512 --config-name=inference_512_test
python3 src/inference.py --config-dir=configs/compare_hidden_dims/512 --config-name=inference_512_train