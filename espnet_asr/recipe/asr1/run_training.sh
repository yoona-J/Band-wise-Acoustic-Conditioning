#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0
export LC_ALL=C

COMMON_ARGS="\
  --ngpu 1 \
  --use_lm false \
  --use_word_lm false \
  --token_type char \
  --train_set train \
  --valid_set dev \
  --test_sets test"

mkdir -p logs

# 1) Conformer
./asr.sh \
  --stage 10 \
  --stop_stage 11 \
  $COMMON_ARGS \
  --asr_config conf/tuning/train_asr_conformer_ctc.yaml \
  --asr_tag conformer_ctc_full \
  2>&1 | tee logs/train_conformer_ctc_$(date +%Y%m%d_%H%M%S).log

# 2) BiLSTM
./asr.sh \
  --stage 11 \
  --stop_stage 11 \
  $COMMON_ARGS \
  --asr_config conf/tuning/train_asr_bilstm_ctc.yaml \
  --asr_tag bilstm_ctc_full \
  2>&1 | tee logs/train_bilstm_ctc_$(date +%Y%m%d_%H%M%S).log

# 3) Transformer
./asr.sh \
  --stage 11 \
  --stop_stage 11 \
  $COMMON_ARGS \
  --asr_config conf/tuning/train_asr_transformer_ctc.yaml \
  --asr_tag transformer_ctc_full \
  2>&1 | tee logs/train_transformer_ctc_$(date +%Y%m%d_%H%M%S).log

# 4) E-Branchformer
./asr.sh \
  --stage 10 \
  --stop_stage 11 \
  $COMMON_ARGS \
  --asr_config conf/tuning/train_asr_ebranchformer_ctc.yaml \
  --asr_tag ebranchformer_ctc_full \
  2>&1 | tee logs/train_ebranchformer_ctc_$(date +%Y%m%d_%H%M%S).log
