#!/usr/bin/env bash
set -e

. ./path.sh
export LC_ALL=C

mkdir -p logs

cut -d' ' -f1 dump/raw/test/wav.scp > dump/raw/test/keys.txt
# if you want to make 1K test predict, please uncomment
# cut -d' ' -f1 dump/raw/test_1k/wav.scp > dump/raw/test_1k/keys.txt

run_decode () {
  GPU=$1
  TAG=$2
  MODEL=$3

  EXP_DIR="exp/asr_${TAG}"
  OUT_DIR="${EXP_DIR}/direct_gpu_test1K"
  LOG_FILE="logs/decode_${TAG}_direct_$(date +%Y%m%d_%H%M%S).log"

  echo "========================================="
  echo "[Decode] ${TAG}"
  echo "GPU=${GPU}"
  echo "CONFIG=${EXP_DIR}/config.yaml"
  echo "MODEL=${EXP_DIR}/${MODEL}"
  echo "OUT=${OUT_DIR}"
  echo "========================================="

  rm -rf "${OUT_DIR}"

  CUDA_VISIBLE_DEVICES=${GPU} python3 -m espnet2.bin.asr_inference \
    --batch_size 1 \
    --ngpu 1 \
    --num_workers 0 \
    --beam_size 1 \
    --ctc_weight 1.0 \
    --data_path_and_name_and_type dump/raw/test/wav.scp,speech,sound \
    --key_file dump/raw/test/keys.txt \
    # if you want to make 1K test predict, please uncomment
    # --data_path_and_name_and_type dump/raw/test_1k/wav.scp,speech,sound \
    # --key_file dump/raw/test_1k/keys.txt \
    --asr_train_config "${EXP_DIR}/config.yaml" \
    --asr_model_file "${EXP_DIR}/${MODEL}" \
    --output_dir "${OUT_DIR}" \
    2>&1 | tee "${LOG_FILE}"
}

# Sequential execution on GPU 1
run_decode 1 transformer_ctc_full valid.acc.ave.pth
run_decode 1 ebranchformer_ctc_full valid.acc.ave.pth

echo "DONE"