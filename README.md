# When the Same Transcript Sounds Different: Diagnosing and Mitigating Recognition Instability in Low-Information Conversational ASR

# Overview

Recent large-scale pretrained ASR models achieve strong average recognition performance, yet hallucination and recognition instability remain persistent in spontaneous conversational speech. We investigate low-information speech as a source of ASR instability, focusing on utterances with limited lexical evidence but substantial acoustic/prosodic variation. Through analyses of Korean regional conversational speech, we show that instability is particularly pronounced when sparse lexical content is distributed over longer acoustic durations, and that identical transcripts can yield diverse ASR predictions under different acoustic realizations. We define this phenomenon as same-transcript instability and analyze its acoustic correlates. To mitigate this failure mode, we propose a lightweight acoustic conditioning module that injects 11 utterance-level acoustic/prosodic features into encoder input representations. Under matched training conditions, acoustic conditioning improves an E-Branchformer hybrid CTC/attention baseline from 7.09\% to 6.00\% CER and reduces hallucination rate from 1.06\% to 0.73\% on the full test set. Additional same-transcript analyses show reduced prediction diversity and hallucination-like outputs, suggesting that acoustic conditioning can stabilize conversational ASR under low-information conditions.

---

# Our Main Contributions

- We define Low-Information Speech as a lexical-acoustic sparsity condition linked to ASR hallucination and recognition instability.
- We introduce same-transcript instability as a controlled diagnostic perspective for analyzing instability under identical lexical content.
- We experimentally show that ASR instability is more strongly associated with lexical-acoustic sparsity and prosodic realization than with dialect region itself.
- We propose a lightweight, analysis-motivated acoustic conditioning (AC) module that mitigates hallucination-like instability under low-information and same-transcript conditions.

---

# Repository Structure

Main code repository: 

`Band-wise_Multi-head_Acoustic_Conditioning/` Typical structure:

```
conf/
espnet2/
local/
scripts/
utils/
```

The repository includes:

- ESPnet-based ASR training/inference code
- Acoustic Conditioning implementation
- preprocessing and evaluation scripts
- selected experiment configurations

Large checkpoints may not be included.

---

# Dataset Repositories

Because of dataset size limitations, the corpus is distributed across multiple dataset repositories.

## 1. Raw Speech Corpus (Baseline + AC)

Dataset: [Raw Datasets](https://zenodo.org/records/20421255)

`/raw` Contents:

```
raw/
├── train
├── dev
├── test
└── test_1k
```

Contains:

- wav.scp
- text
- utt2spk
- speaker metadata
- FLAC audio

Used by:

- Encoder Baseline models
- Encoder + decoder only models

---

## 2. Acoustic Conditioning Features (Main AC)

Dataset: [Acoustic Conditioning Features Datasets](https://zenodo.org/records/20421255)

`/hallu_acoustic_feats` Contents:

```
hallu_acoustic_feats/
├── train
├── dev
├── test
└── test_1k
```

Contains:

- acoustic_feats.scp
- utterance-level acoustic features (.npy)

Used by:

- Acoustic Conditioning (main model)

---

## 3. Temporal Acoustic Features (Ablation)

Dataset: [Temporal Acoustic Features Datasets](https://zenodo.org/records/20421255)

`/hallu_acoustic_feats_temporal` Used for:

- Ablation experiments only
- Contains temporal acoustic features.

Note: `test_1k` references test/norm features via relative paths.

---

## 4. Voice Quality Features (Ablation)

Dataset: [Voice Quality Features](https://zenodo.org/records/20421255)

`/hallu_acoustic_feats_voice` Used for:

- Ablation experiments only
- Contains voice-quality feature variants.

Note: `test_1k` references test/norm features via relative paths.

---

# Dataset Statistics

After preprocessing and filtering:

| Split | Utterances | Ratio |
|---|---:|---:|
| Train | 1,283,280 | 69.6% |
| Validation | 179,058 | 9.7% |
| Test | 381,816 | 20.7% |
| Total | 1,844,154 | 100% |

Test subset:

| Category | Count |
|---|---:|
| Low-Information | 121,705 |
| Normal | 260,111 |

Corpus coverage:

- 4,511 speakers
- multiple Korean regional varieties
- 635 same-transcript groups
- 12,605 same-transcript utterances

The corpus is limited to Korean conversational speech but provides broad regional and acoustic diversity suitable for studying recognition instability.

---

# Low-Information Speech

We define Low-Information Speech using: `duration ≤ 2 sec OR token count ≤ 2`

This definition captures:

- short utterances
- fillers
- hesitation
- elongated speech
- lexically sparse conversational speech

<!-- <img width="3921" height="1591" alt="figure1" src="https://github.com/user-attachments/assets/27f846a0-7ac3-4734-ba35-d3d2ea7e4250" /> -->
<img width="100%" alt="figure1" src="./figure1.png" />
<p align='center'>Figure 1. Examples of Recognition Instability under Low-Information Speech.</p>


---

# Acoustic Conditioning

The proposed framework combines:

<!-- <img width="5893" height="2117" alt="figure5" src="https://github.com/user-attachments/assets/ef31fd9b-8f26-4a50-8784-fe6e99ca68ba" /> -->
<img width="100%" alt="figure5" src="./figure5.png" />
<p align='center'>Figure 5. Overall architecture of the proposed framework.</p>


Acoustic Conditioning injects:

- F0 std
- F0 range
- token/sec
- duration/token
- silence ratio
- number of pauses
- mean pause duration
- RMS energy
- HNR
- jitter
- shimmer

via residual conditioning and gating.

---

# Experimental Results

Primary comparison:

| Model | CER | Hall. |
|---|---:|---:|
| E-Branch + Dec | 7.09% | 1.06% |
| AC + E-Branch + Dec | **6.00%** | **0.73%** |

Acoustic Conditioning reduced:

- CER
- Weighted-CER
- hallucination rate
- same-transcript prediction diversity

---

# Training and Reproducibility

## Baseline Models

The repository provides baseline training and decoding scripts for:

- BiLSTM
- Transformer
- Conformer
- E-Branchformer

Training: `./run_training.sh`

Test decoding: `./run_decode1.sh`, `./run_decode2.sh`

Baseline models use: `/raw` dataset only.

---

# E-Branchformer + Transformer Decoder

Uses: `/raw` dataset only.

Training: 

```
CUDA_VISIBLE_DEVICES=1 ./asr.sh \
  --stage 11 \
  --stop_stage 11 \
  --ngpu 1 \
  --use_lm false \
  --use_word_lm false \
  --token_type char \
  --train_set train \
  --valid_set dev \
  --test_sets test \
  --asr_config conf/tuning/train_asr_ebranchformer_fix_hybrid.yaml \
  --asr_tag ebranchformer_fix_hybrid \
  2>&1 | tee logs/train_ebranchformer_fix_hybrid_$(date +%Y%m%d_%H%M%S).log
```

Test:

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1 python3 -m espnet2.bin.asr_inference \
--batch_size 1 \
--ngpu 1 \
--data_path_and_name_and_type dump/raw/test/wav.scp,speech,sound \
--key_file dump/raw/test/keys.txt \
--asr_train_config exp/asr_ebranchformer_fix_hybrid/config.yaml \
--asr_model_file exp/asr_ebranchformer_fix_hybrid/valid.loss.best.pth \
--ctc_weight 0.3 \
--beam_size 5 \
--maxlenratio 1.0 \
--output_dir exp/asr_ebranchformer_fix_hybrid/direct_gpu_test \
2>&1 | tee logs/decode_ebranchformer_fix_hybrid_$(date +%Y%m%d_%H%M%S).log
```

1k Subset Test:

```
CUDA_VISIBLE_DEVICES=1 python3 -m espnet2.bin.asr_inference \
--batch_size 1 \
--ngpu 1 \
--data_path_and_name_and_type dump/raw/test_1k/wav.scp,speech,sound \
--key_file dump/raw/test_1k/keys.txt \
--asr_train_config exp/asr_ebranchformer_fix_hybrid/config.yaml \
--asr_model_file exp/asr_ebranchformer_fix_hybrid/valid.loss.best.pth \
--ctc_weight 0.3 \
--beam_size 5 \
--maxlenratio 1.0 \
--output_dir exp/asr_ebranchformer_hybrid/direct_gpu_test1k \
2>&1 | tee logs/decode_ebranchformer_fix_hybrid_test1k_$(date +%Y%m%d_%H%M%S).log
```

Note: If you want to Test of Training, make sure move the dataset in `/espnet_asr/recipe/asr1/dump` (ex. `/espnet_asr/recipe/asr1/dump/raw`)


---

# Acoustic Conditioning + E-Branchformer + CTC only

Uses: `/hallu_acoustic_feats` dataset only.

Collect stats:

```
CUDA_VISIBLE_DEVICES=0 python3 -m espnet2.bin.asr_train \
  --collect_stats true \
  --allow_variable_data_keys true \
  --use_preprocessor true \
  --bpemodel none \
  --token_type char \
  --token_list data/token_list/char/tokens.txt \
  --non_linguistic_symbols none \
  --cleaner none \
  --g2p none \
  --train_data_path_and_name_and_type dump/raw/train/wav.scp,speech,sound \
  --train_data_path_and_name_and_type dump/raw/train/text,text,text \
  --train_data_path_and_name_and_type dump/hallu_acoustic_feats/train/acoustic_feats.scp,acoustic_feat,npy \
  --valid_data_path_and_name_and_type dump/raw/dev/wav.scp,speech,sound \
  --valid_data_path_and_name_and_type dump/raw/dev/text,text,text \
  --valid_data_path_and_name_and_type dump/hallu_acoustic_feats/dev/acoustic_feats.scp,acoustic_feat,npy \
  --config conf/tuning/train_asr_ebranchformer_fix_hybrid_hallu.yaml \
  --output_dir exp/asr_stats_ebranchformer_fix_hybrid_hallu \
  --ngpu 1 \
  2>&1 | tee logs/collect_stats_ebranchformer_fix_hybrid_hallu_$(date +%Y%m%d_%H%M%S).log
```

Training:

```
CUDA_VISIBLE_DEVICES=1 python3 -m espnet2.bin.asr_train \
  --allow_variable_data_keys true \
  --use_preprocessor true \
  --bpemodel none \
  --token_type char \
  --token_list data/token_list/char/tokens.txt \
  --non_linguistic_symbols none \
  --cleaner none \
  --g2p none \
  --train_data_path_and_name_and_type dump/raw/train/wav.scp,speech,sound \
  --train_data_path_and_name_and_type dump/raw/train/text,text,text \
  --train_data_path_and_name_and_type dump/hallu_acoustic_feats/train/acoustic_feats.scp,acoustic_feat,npy \
  --valid_data_path_and_name_and_type dump/raw/dev/wav.scp,speech,sound \
  --valid_data_path_and_name_and_type dump/raw/dev/text,text,text \
  --valid_data_path_and_name_and_type dump/hallu_acoustic_feats/dev/acoustic_feats.scp,acoustic_feat,npy \
  --train_shape_file exp/asr_stats_ebranchformer_fix_hybrid_hallu/train/speech_shape \
  --train_shape_file exp/asr_stats_ebranchformer_fix_hybrid_hallu/train/text_shape \
  --train_shape_file exp/asr_stats_ebranchformer_fix_hybrid_hallu/train/acoustic_feat_shape \
  --valid_shape_file exp/asr_stats_ebranchformer_fix_hybrid_hallu/valid/speech_shape \
  --valid_shape_file exp/asr_stats_ebranchformer_fix_hybrid_hallu/valid/text_shape \
  --valid_shape_file exp/asr_stats_ebranchformer_fix_hybrid_hallu/valid/acoustic_feat_shape \
  --config conf/tuning/train_asr_ebranchformer_fix_ctc_hallu.yaml \
  --output_dir exp/asr_ebranchformer_fix_hallu \
  --ngpu 1 \
  2>&1 | tee logs/train_ebranchformer_fix_hallu_$(date +%Y%m%d_%H%M%S).log
```

Test:

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1 python3 -m espnet2.bin.asr_inference \
--allow_variable_data_keys true \
--batch_size 1 \
--ngpu 1 \
--data_path_and_name_and_type dump/raw/test/wav.scp,speech,sound \
--data_path_and_name_and_type dump/hallu_acoustic_feats/test/acoustic_feats.scp,acoustic_feat,npy \
--key_file dump/raw/test/keys.txt \
--asr_train_config exp/asr_ebranchformer_fix_hallu/config.yaml \
--asr_model_file exp/asr_ebranchformer_fix_hallu/valid.acc.ave.pth \
--beam_size 1 \
--ctc_weight 1.0 \
--maxlenratio 1.0 \
--output_dir exp/asr_ebranchformer_fix_hallu/direct_gpu_test \
2>&1 | tee logs/asr_ebranchformer_fix_hallu$(date +%Y%m%d_%H%M%S).log
```

1k Subset Test:

```
CUDA_VISIBLE_DEVICES=1 python3 -m espnet2.bin.asr_inference \
--allow_variable_data_keys true \
--batch_size 1 \
--ngpu 1 \
--data_path_and_name_and_type dump/raw/test_1k/wav.scp,speech,sound \
--data_path_and_name_and_type dump/hallu_acoustic_feats/test_1k/acoustic_feats.scp,acoustic_feat,npy \
--key_file dump/raw/test_1k/keys.txt \
--asr_train_config exp/asr_ebranchformer_fix_hallu/config.yaml \
--asr_model_file exp/asr_ebranchformer_fix_hallu/valid.loss.best.pth \
--ctc_weight 1.0 \
--beam_size 1 \
--maxlenratio 1.0 \
--output_dir exp/asr_ebranchformer_fix_hallu/direct_gpu_test1k \
2>&1 | tee logs/asr_ebranchformer_fix_hallu_1k_$(date +%Y%m%d_%H%M%S).log
```

Note: If you want to Test of Training, make sure move the dataset in `/espnet_asr/recipe/asr1/dump` (ex. `/espnet_asr/recipe/asr1/dump/hallu_acoustic_feats`)

---

# Acoustic Conditioning + E-Branchformer + Transformer Decoder

Uses: `/hallu_acoustic_feats`

Training:

```
CUDA_VISIBLE_DEVICES=0 python3 -m espnet2.bin.asr_train \
--allow_variable_data_keys true \
--use_preprocessor true \
--bpemodel none \
--token_type char \
--token_list data/token_list/char/tokens.txt \
--non_linguistic_symbols none \
--cleaner none \
--g2p none \
--train_data_path_and_name_and_type dump/raw/train/wav.scp,speech,sound \
--train_data_path_and_name_and_type dump/raw/train/text,text,text \
--train_data_path_and_name_and_type dump/hallu_acoustic_feats/train/acoustic_feats.scp,acoustic_feat,npy \
--valid_data_path_and_name_and_type dump/raw/dev/wav.scp,speech,sound \
--valid_data_path_and_name_and_type dump/raw/dev/text,text,text \
--valid_data_path_and_name_and_type dump/hallu_acoustic_feats/dev/acoustic_feats.scp,acoustic_feat,npy \
--train_shape_file exp/asr_stats_ebranchformer_fix_hybrid_hallu/train/speech_shape \
--train_shape_file exp/asr_stats_ebranchformer_fix_hybrid_hallu/train/text_shape \
--train_shape_file exp/asr_stats_ebranchformer_fix_hybrid_hallu/train/acoustic_feat_shape \
--valid_shape_file exp/asr_stats_ebranchformer_fix_hybrid_hallu/valid/speech_shape \
--valid_shape_file exp/asr_stats_ebranchformer_fix_hybrid_hallu/valid/text_shape \
--valid_shape_file exp/asr_stats_ebranchformer_fix_hybrid_hallu/valid/acoustic_feat_shape \
--config conf/tuning/train_asr_ebranchformer_fix_hybrid_hallu.yaml \
--output_dir exp/asr_ebranchformer_fix_hybrid_hallu \
--ngpu 1 \
2>&1 | tee logs/train_ebranchformer_fix_hybrid_hallu_$(date +%Y%m%d_%H%M%S).log
```

Test:

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 python3 -m espnet2.bin.asr_inference \
--allow_variable_data_keys true \
--batch_size 1 \
--ngpu 1 \
--data_path_and_name_and_type dump/raw/test/wav.scp,speech,sound \
--data_path_and_name_and_type dump/hallu_acoustic_feats/test/acoustic_feats.scp,acoustic_feat,npy \
--key_file dump/raw/test/keys.txt \
--asr_train_config exp/asr_ebranchformer_fix_hybrid_hallu/config.yaml \
--asr_model_file exp/asr_ebranchformer_fix_hybrid_hallu/valid.acc.ave.pth \
--beam_size 5 \
--ctc_weight 0.3 \
--maxlenratio 1.0 \
--output_dir exp/asr_ebranchformer_fix_hybrid_hallu/direct_gpu_test \
2>&1 | tee logs/decode_ebranchformer_fix_hybrid_hallu_test_$(date +%Y%m%d_%H%M%S).log
```

1k subset test:

```
CUDA_VISIBLE_DEVICES=0 python3 -m espnet2.bin.asr_inference \
  --allow_variable_data_keys true \
  --batch_size 1 \
  --ngpu 1 \
  --data_path_and_name_and_type dump/raw/test_1k/wav.scp,speech,sound \
  --data_path_and_name_and_type dump/hallu_acoustic_feats/test_1k/acoustic_feats.scp,acoustic_feat,npy \
  --key_file dump/raw/test_1k/keys.txt \
  --asr_train_config exp/asr_ebranchformer_fix_hybrid_hallu/config.yaml \
  --asr_model_file exp/asr_ebranchformer_fix_hybrid_hallu/valid.acc.ave.pth \
  --beam_size 5 \
  --ctc_weight 0.3 \
  --maxlenratio 1.0 \
  --output_dir exp/asr_ebranchformer_fix_hybrid_hallu/direct_gpu_test1k \
  2>&1 | tee logs/asr_ebranchformer_fix_hybrid_hallu_test1k_$(date +%Y%m%d_%H%M%S).log
```

Note: If you want to Test of Training, make sure move the dataset in `/espnet_asr/recipe/asr1/dump` (ex. `/espnet_asr/recipe/asr1/dump/hallu_acoustic_feats`)


---

# test_1k Subset

`test_1k` is a 1,000-utterance sampled validation subset used for preliminary experiments and validation analyses reported in the paper.

---

# Limitations

The present dataset is limited to Korean conversational speech. However, hallucination and recognition instability have also been reported across other languages and ASR systems, suggesting that the proposed framework and diagnostic setting may support future cross-lingual and cross-domain validation.

