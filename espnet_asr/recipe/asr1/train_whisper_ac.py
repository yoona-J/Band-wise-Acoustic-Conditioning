import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import whisper
from whisper.audio import HOP_LENGTH, N_FRAMES, N_SAMPLES
from whisper.tokenizer import get_tokenizer

try:
    from whisper_multihead_ac import AcousticConditioning
except ImportError:
    from whisper_multihead_ac import (
        MultiHeadAcousticConditioning as AcousticConditioning,
    )


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# SCP / text loading
# ============================================================

def read_key_value(path):
    data = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) == 1:
                data[parts[0]] = ""
            else:
                data[parts[0]] = parts[1]

    return data


def get_valid_mel_frames(num_samples):
    """Return the number of non-padding Whisper Mel frames.

    Whisper pads/trims every training utterance to N_SAMPLES (30 s).
    AC should be active only over the portion originating from the
    original waveform, not over the zero-padded tail.  Ceil is used so
    that a final partial hop containing real audio is kept active.
    """

    valid_samples = min(
        int(num_samples),
        int(N_SAMPLES),
    )

    if valid_samples <= 0:
        return 0

    valid_frames = (
        valid_samples
        + int(HOP_LENGTH)
        - 1
    ) // int(HOP_LENGTH)

    return min(
        int(valid_frames),
        int(N_FRAMES),
    )


# ============================================================
# Dataset
# ============================================================

class WhisperACDataset(Dataset):

    def __init__(
        self,
        wav_scp,
        text_path,
        acoustic_scp,
        tokenizer,
        n_text_ctx,
        max_items=None,
    ):

        self.wav_map = read_key_value(wav_scp)
        self.text_map = read_key_value(text_path)
        self.acoustic_map = read_key_value(acoustic_scp)

        # Preserve wav.scp order
        keys = [
            k for k in self.wav_map
            if (
                k in self.text_map
                and k in self.acoustic_map
            )
        ]

        if max_items is not None:
            keys = keys[:max_items]

        self.keys = keys
        self.tokenizer = tokenizer
        self.n_text_ctx = n_text_ctx

        print(
            f"[Dataset] wav={len(self.wav_map)} "
            f"text={len(self.text_map)} "
            f"acoustic={len(self.acoustic_map)} "
            f"matched={len(self.keys)}"
        )

        if len(self.keys) == 0:
            raise RuntimeError("No matched utterances.")


    def __len__(self):
        return len(self.keys)


    def __getitem__(self, idx):

        utt_id = self.keys[idx]

        wav_path = self.wav_map[utt_id]
        text = self.text_map[utt_id]
        acoustic_path = self.acoustic_map[utt_id]

        # ----------------------------------------------------
        # Whisper audio preprocessing
        # ----------------------------------------------------

        audio = whisper.load_audio(wav_path)

        # Number of Mel frames that originate from the real waveform.
        # Frames after this point are Whisper's fixed 30-s zero padding.
        valid_mel_frames = get_valid_mel_frames(
            len(audio)
        )

        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio)

        # [80, 3000]
        mel = mel.float()

        # ----------------------------------------------------
        # Acoustic feature
        # ----------------------------------------------------

        acoustic_feat = np.load(acoustic_path).astype(
            np.float32
        )

        if acoustic_feat.shape != (9,):
            raise ValueError(
                f"{utt_id}: expected acoustic feat (9,), "
                f"got {acoustic_feat.shape}"
            )

        acoustic_feat = torch.from_numpy(
            acoustic_feat
        )

        # ----------------------------------------------------
        # Whisper teacher-forcing tokens
        # ----------------------------------------------------

        prefix = list(
            self.tokenizer.sot_sequence_including_notimestamps
        )

        text_tokens = self.tokenizer.encode(
            text
        )

        # sequence:
        #
        # <sot> <ko> <transcribe> <notimestamps>
        # text ...
        # <eot>
        #
        max_text_tokens = (
            self.n_text_ctx
            - len(prefix)
            - 1
        )

        text_tokens = text_tokens[
            :max_text_tokens
        ]

        full = (
            prefix
            + text_tokens
            + [self.tokenizer.eot]
        )

        tokens_in = torch.tensor(
            full[:-1],
            dtype=torch.long,
        )

        targets = torch.tensor(
            full[1:],
            dtype=torch.long,
        )

        return {
            "utt_id": utt_id,
            "mel": mel,
            "acoustic_feat": acoustic_feat,
            "valid_mel_frames": valid_mel_frames,
            "tokens_in": tokens_in,
            "targets": targets,
        }


# ============================================================
# Batch
# ============================================================

def collate_fn(batch):

    mel = torch.stack(
        [x["mel"] for x in batch],
        dim=0,
    )

    acoustic_feat = torch.stack(
        [x["acoustic_feat"] for x in batch],
        dim=0,
    )

    valid_mel_frames = torch.tensor(
        [x["valid_mel_frames"] for x in batch],
        dtype=torch.long,
    )

    max_len = max(
        x["tokens_in"].numel()
        for x in batch
    )

    tokens_in = torch.full(
        (len(batch), max_len),
        fill_value=0,
        dtype=torch.long,
    )

    targets = torch.full(
        (len(batch), max_len),
        fill_value=-100,
        dtype=torch.long,
    )

    for i, x in enumerate(batch):

        n = x["tokens_in"].numel()

        tokens_in[i, :n] = x[
            "tokens_in"
        ]

        targets[i, :n] = x[
            "targets"
        ]

    return {
        "utt_id": [
            x["utt_id"]
            for x in batch
        ],
        "mel": mel,
        "acoustic_feat": acoustic_feat,
        "valid_mel_frames": valid_mel_frames,
        "tokens_in": tokens_in,
        "targets": targets,
    }


# ============================================================
# Whisper encoder wrapper
#
# Four AC locations are supported:
#
# 1) postconv (legacy experiment)
#    [B,80,T] -> Whisper conv frontend -> [B,T,D] -> AC
#
# 2) postconv_stage2 (two-stage shared-gamma calibration)
#    Stage-1 postconv AC frozen; one shared gamma learned.
#
# 3) bandwise_bridge (legacy frequency-band bridge experiment)
#    [B,80,T]
#      ├─> frozen Whisper conv frontend ----------------> H_base
#      └─> band-wise AC on 80 log-Mel bins -> frontend -> H_ac
#
#    H = H_base + lambda * (H_ac - H_base)
#
# 4) bandwise_adapter (frequency-band residual adapter)
#    [B,80,T] ------------------------------------------> H_base
#       |
#       +-> band-wise AC on 80 log-Mel bins -> Delta X
#             -> grouped trainable adapter -> Delta H
#
#    H = H_base + lambda_adapter * Delta H
#
# 5) bandwise_frontend (AC + Conv1 adaptation)
#    [B,80,T] -> band-wise AC on 80 log-Mel bins -> X'
#             -> Whisper Conv1 (TRAINABLE, small LR)
#             -> Whisper Conv2 (FROZEN) -> Transformer
#
# 6) bandwise_frozen (pure AC -> frozen Whisper)
#    [B,80,T] -> band-wise AC on 80 log-Mel bins -> X'
#             -> original Whisper frontend/encoder/decoder (ALL FROZEN)
#    Only the AC module is trainable.
#
# Frequency-band modes preserve the original AC inductive bias:
# - the SAME 9-D acoustic descriptor is fed to all four heads;
# - only the 80-D log-Mel feature axis is split into four 20-bin bands;
# - each head modifies only its corresponding frequency band;
# - the frozen Whisper frontend always receives the ORIGINAL Mel input.
# ============================================================

class AcousticConditionedWhisperEncoder(nn.Module):

    def __init__(
        self,
        base_encoder,
        acoustic_dim=9,
        hidden_dim=128,
        dropout=0.1,
        ac_location="postconv",
        bridge_init=0.1,
        adapter_hidden=128,
        adapter_scale_init=0.01,
        adapter_scale_max=0.1,
        calibration_init=0.10,
        calibration_max=0.20,
    ):

        super().__init__()

        if ac_location not in {
            "postconv",
            "postconv_stage2",
            "bandwise_bridge",
            "bandwise_adapter",
            "bandwise_frontend",
            "bandwise_frozen",
        }:
            raise ValueError(
                f"Unknown ac_location: {ac_location}"
            )

        self.base_encoder = base_encoder
        self.ac_location = ac_location
        self.current_acoustic_feat = None
        self.current_valid_mel_frames = None

        # Legacy post-conv global calibration. Kept only so old
        # experiments/checkpoints can still be reproduced.
        self.conditioning_beta = 1.0

        self.adapter_hidden = int(adapter_hidden)
        self.adapter_scale_max = float(adapter_scale_max)
        self.calibration_max = float(calibration_max)

        # ----------------------------------------------------
        # Frequency-band modes: AC sees ORIGINAL 80 log-Mel bins.
        # The same 9-D descriptor enters all four heads.
        # Only X is split into four contiguous 20-bin bands.
        # ----------------------------------------------------
        if self.ac_location in {
            "bandwise_bridge",
            "bandwise_adapter",
            "bandwise_frontend",
            "bandwise_frozen",
        }:
            self.acoustic_input_dim = 80

            self.acoustic_conditioning = AcousticConditioning(
                acoustic_dim=acoustic_dim,
                input_dim=80,
                hidden_dim=hidden_dim,
                dropout_rate=dropout,
                num_heads=4,
            )
        else:
            # Legacy post-conv AC. For Whisper-small this is 768.
            self.acoustic_input_dim = int(
                base_encoder.positional_embedding.shape[-1]
            )

            self.acoustic_conditioning = AcousticConditioning(
                acoustic_dim=acoustic_dim,
                input_dim=self.acoustic_input_dim,
                hidden_dim=hidden_dim,
                dropout_rate=dropout,
                num_heads=4,
            )

        # ----------------------------------------------------
        # Legacy bandwise_bridge lambda in (0, 1).
        # ----------------------------------------------------
        if self.ac_location == "bandwise_bridge":
            if not (0.0 < bridge_init < 1.0):
                raise ValueError(
                    "bridge_init must be between 0 and 1."
                )

            bridge_logit_init = math.log(
                bridge_init / (1.0 - bridge_init)
            )

            self.bridge_logit = nn.Parameter(
                torch.tensor(
                    bridge_logit_init,
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter(
                "bridge_logit",
                None,
            )

        # ----------------------------------------------------
        # NEW bandwise_adapter path.
        #
        # Delta X: [B,80,T]
        #   -> grouped Conv1d, groups=4
        #      Each group sees exactly one 20-bin frequency band.
        #   -> GELU
        #   -> 1x1 projection to Whisper hidden dimension D
        #
        # The first grouped convolution preserves the four-band
        # structure before the final projection mixes the band
        # information into Whisper's latent representation.
        # ----------------------------------------------------
        if self.ac_location == "bandwise_adapter":
            if self.adapter_hidden % 4 != 0:
                raise ValueError(
                    "adapter_hidden must be divisible by 4 so "
                    "groups=4 preserves the four frequency bands."
                )

            if not (
                0.0 < adapter_scale_init < adapter_scale_max
            ):
                raise ValueError(
                    "adapter_scale_init must satisfy "
                    "0 < init < adapter_scale_max."
                )

            whisper_dim = int(
                base_encoder.positional_embedding.shape[-1]
            )

            self.ac_adapter = nn.Sequential(
                nn.Conv1d(
                    in_channels=80,
                    out_channels=self.adapter_hidden,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=4,
                ),
                nn.GELU(),
                nn.Conv1d(
                    in_channels=self.adapter_hidden,
                    out_channels=whisper_dim,
                    kernel_size=1,
                ),
            )

            # Start EXACTLY from the plain-Whisper representation.
            # At initialization Delta H = 0, so H = H_base.
            nn.init.zeros_(self.ac_adapter[-1].weight)
            if self.ac_adapter[-1].bias is not None:
                nn.init.zeros_(self.ac_adapter[-1].bias)

            # lambda_adapter = max_scale * sigmoid(logit)
            # This keeps the latent residual small and bounded.
            ratio = (
                adapter_scale_init
                / self.adapter_scale_max
            )
            adapter_logit_init = math.log(
                ratio / (1.0 - ratio)
            )

            self.adapter_logit = nn.Parameter(
                torch.tensor(
                    adapter_logit_init,
                    dtype=torch.float32,
                )
            )
        else:
            self.ac_adapter = None
            self.register_parameter(
                "adapter_logit",
                None,
            )

        # ----------------------------------------------------
        # Stage-2 post-conv calibration.
        # Stage-1 AC residual networks/gates are frozen; their
        # internal alpha_i are normalized to 1.0 and one shared
        # bounded gamma is learned.
        # ----------------------------------------------------
        if self.ac_location == "postconv_stage2":
            if not (0.0 < calibration_init < calibration_max):
                raise ValueError(
                    "calibration_init must satisfy "
                    "0 < init < calibration_max."
                )
            ratio = calibration_init / calibration_max
            calibration_logit_init = math.log(
                ratio / (1.0 - ratio)
            )
            self.calibration_logit = nn.Parameter(
                torch.tensor(
                    calibration_logit_init,
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter(
                "calibration_logit",
                None,
            )


    @property
    def bridge_scale(self):
        """Learned lambda for legacy bandwise_bridge."""

        if self.bridge_logit is None:
            return None

        return torch.sigmoid(
            self.bridge_logit
        )


    @property
    def adapter_scale(self):
        """Bounded latent-residual scale for bandwise_adapter."""

        if self.adapter_logit is None:
            return None

        return (
            self.adapter_scale_max
            * torch.sigmoid(
                self.adapter_logit
            )
        )


    @property
    def calibration_scale(self):
        """Shared bounded gamma for postconv_stage2."""
        if self.calibration_logit is None:
            return None
        return (
            self.calibration_max
            * torch.sigmoid(self.calibration_logit)
        )


    def set_acoustic_feat(
        self,
        acoustic_feat,
        valid_mel_frames=None,
    ):
        self.current_acoustic_feat = acoustic_feat
        self.current_valid_mel_frames = valid_mel_frames


    def clear_acoustic_feat(self):
        self.current_acoustic_feat = None
        self.current_valid_mel_frames = None


    def set_conditioning_beta(
        self,
        beta,
    ):
        self.conditioning_beta = float(beta)


    def _run_frontend(self, mel):
        """Run Whisper's convolutional frontend.

        In bandwise_frontend mode Conv1 is trainable while Conv2 stays
        frozen. In every other mode both convolutional layers remain frozen.
        """

        x = F.gelu(
            self.base_encoder.conv1(mel)
        )

        x = F.gelu(
            self.base_encoder.conv2(x)
        )

        return x


    def _bandwise_conditioned_mel(
        self,
        mel,
        acoustic_feat,
    ):
        """Apply AC to the real 80-bin log-Mel frequency axis.

        mel: [B,80,T]
        acoustic_feat: [B,9]

        The SAME [B,9] descriptor is supplied to every AC head.
        AcousticConditioning itself chunks only the last feature
        axis of [B,T,80] into 4 x [B,T,20].
        """

        mel_bt80 = mel.transpose(
            1,
            2,
        )

        conditioned_bt80 = (
            self.acoustic_conditioning(
                mel_bt80,
                acoustic_feat,
            )
        )

        # ----------------------------------------------------
        # Padding-aware AC mask
        # ----------------------------------------------------
        # Whisper pads every utterance to a fixed 30-s Mel tensor.
        # The utterance-level AC residual must not be broadcast over
        # that artificial padded tail.  Keep the original Mel values
        # exactly unchanged after each utterance's valid frame count.
        if self.current_valid_mel_frames is not None:
            valid_frames = (
                self.current_valid_mel_frames
                .to(
                    device=mel_bt80.device,
                    dtype=torch.long,
                )
                .clamp(
                    min=0,
                    max=mel_bt80.size(1),
                )
            )

            if valid_frames.dim() == 0:
                valid_frames = valid_frames.unsqueeze(0)

            if valid_frames.numel() != mel_bt80.size(0):
                raise RuntimeError(
                    "valid_mel_frames batch mismatch: "
                    f"got {valid_frames.numel()}, "
                    f"expected {mel_bt80.size(0)}"
                )

            frame_index = torch.arange(
                mel_bt80.size(1),
                device=mel_bt80.device,
            ).view(1, -1, 1)

            valid_mask = (
                frame_index
                < valid_frames.view(-1, 1, 1)
            ).to(
                dtype=mel_bt80.dtype
            )

            conditioned_bt80 = (
                mel_bt80
                + valid_mask
                * (conditioned_bt80 - mel_bt80)
            )

        return conditioned_bt80.transpose(
            1,
            2,
        )


    def forward(self, mel):

        # ----------------------------------------------------
        # If AC is disabled, use original Whisper encoder
        # exactly as-is.
        # ----------------------------------------------------
        if self.current_acoustic_feat is None:
            return self.base_encoder(mel)

        acoustic_feat = (
            self.current_acoustic_feat
            .to(
                device=mel.device,
                dtype=mel.dtype,
            )
        )

        # ====================================================
        # NEW: true frequency-band AC -> Conv1 adaptation
        #
        # AC operates on the real 80-bin log-Mel axis. The SAME 9-D
        # acoustic descriptor is supplied to all four heads, while only
        # X is split into four contiguous 20-bin frequency bands.
        #
        # The conditioned Mel is then consumed by Whisper's frontend.
        # Conv1 is trainable with a small LR in this mode; Conv2 and the
        # remaining Whisper encoder/decoder stay frozen.
        # ====================================================
        if self.ac_location in {"bandwise_frontend", "bandwise_frozen"}:

            conditioned_mel = (
                self._bandwise_conditioned_mel(
                    mel,
                    acoustic_feat,
                )
            )

            x = self._run_frontend(
                conditioned_mel
            )

            # [B,D,T'] -> [B,T',D]
            x = x.permute(
                0,
                2,
                1,
            )

        # ====================================================
        # NEW: true frequency-band AC -> residual adapter
        # ====================================================
        elif self.ac_location == "bandwise_adapter":

            # Whisper's frozen frontend always sees ORIGINAL Mel.
            # [B,80,T] -> [B,D,T']
            h_base = self._run_frontend(
                mel
            )

            # AC itself operates only on the original 80-bin
            # frequency axis. Same 9-D descriptor goes to all heads.
            conditioned_mel = (
                self._bandwise_conditioned_mel(
                    mel,
                    acoustic_feat,
                )
            )

            # Keep only the AC-induced Mel residual.
            # [B,80,T]
            delta_mel = (
                conditioned_mel
                - mel
            )

            # Grouped adapter:
            # [B,80,T] -> [B,D,T']
            delta_h = self.ac_adapter(
                delta_mel
            )

            if delta_h.shape != h_base.shape:
                raise RuntimeError(
                    "bandwise_adapter shape mismatch: "
                    f"delta_h={delta_h.shape}, "
                    f"h_base={h_base.shape}"
                )

            scale = self.adapter_scale.to(
                device=h_base.device,
                dtype=h_base.dtype,
            )

            # Plain Whisper is the anchor; AC enters only as a
            # small learned latent residual.
            h = (
                h_base
                + scale * delta_h
            )

            # [B,D,T'] -> [B,T',D]
            x = h.permute(
                0,
                2,
                1,
            )

        # ====================================================
        # Legacy: true frequency-band AC + parallel Whisper bridge
        # ====================================================
        elif self.ac_location == "bandwise_bridge":

            h_base = self._run_frontend(
                mel
            )

            conditioned_mel = (
                self._bandwise_conditioned_mel(
                    mel,
                    acoustic_feat,
                )
            )

            # Same frozen Whisper frontend for conditioned Mel.
            h_ac = self._run_frontend(
                conditioned_mel
            )

            bridge = self.bridge_scale.to(
                device=h_base.device,
                dtype=h_base.dtype,
            )

            h = (
                h_base
                + bridge
                * (h_ac - h_base)
            )

            x = h.permute(
                0,
                2,
                1,
            )

        # ====================================================
        # Legacy: post-conv AC, kept for reproducibility
        # ====================================================
        else:

            x = self._run_frontend(
                mel
            )

            x = x.permute(
                0,
                2,
                1,
            )

            x_original = x

            x_conditioned = (
                self.acoustic_conditioning(
                    x,
                    acoustic_feat,
                )
            )

            if self.ac_location == "postconv_stage2":
                gamma = self.calibration_scale.to(
                    device=x_original.device,
                    dtype=x_original.dtype,
                )
                x = (
                    x_original
                    + gamma
                    * (x_conditioned - x_original)
                )
            else:
                x = (
                    x_original
                    + self.conditioning_beta
                    * (x_conditioned - x_original)
                )

        # Same check used by Whisper AudioEncoder.
        assert (
            x.shape[1:]
            == self.base_encoder.positional_embedding.shape
        ), (
            "incorrect audio shape: "
            f"{x.shape[1:]} vs "
            f"{self.base_encoder.positional_embedding.shape}"
        )

        # ----------------------------------------------------
        # Original Whisper encoder continues from here
        # ----------------------------------------------------
        x = (
            x
            + self.base_encoder.positional_embedding
        ).to(x.dtype)

        for block in self.base_encoder.blocks:
            x = block(x)

        x = self.base_encoder.ln_post(x)

        return x


# ============================================================
# AC logging helpers
# ============================================================

def get_alpha_values(model):
    return [
        float(a.detach().cpu())
        for a in model.encoder.acoustic_conditioning.alphas
    ]


def get_alpha_mean(model):
    values = get_alpha_values(model)
    return sum(values) / len(values)


# ============================================================
# Validation loss
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(
        loader,
        desc="Validation",
        leave=False,
    ):

        mel = batch["mel"].to(
            device,
            non_blocking=True,
        )

        acoustic_feat = (
            batch["acoustic_feat"]
            .to(
                device,
                non_blocking=True,
            )
        )

        valid_mel_frames = (
            batch["valid_mel_frames"]
            .to(
                device,
                non_blocking=True,
            )
        )

        tokens_in = (
            batch["tokens_in"]
            .to(
                device,
                non_blocking=True,
            )
        )

        targets = (
            batch["targets"]
            .to(
                device,
                non_blocking=True,
            )
        )

        model.encoder.set_acoustic_feat(
            acoustic_feat,
            valid_mel_frames=valid_mel_frames,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(
                device.type == "cuda"
            ),
        ):

            logits = model(
                mel,
                tokens_in,
            )

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )

        n_tokens = (
            targets
            != -100
        ).sum().item()

        total_loss += loss.item()
        total_tokens += n_tokens

    model.encoder.clear_acoustic_feat()

    return (
        total_loss
        / max(total_tokens, 1)
    )

# ============================================================
# Test decoding
# ============================================================

@torch.no_grad()
def run_test(
    model,
    device,
    checkpoint_path,
    wav_scp,
    acoustic_scp,
    output_dir,
    test_alpha=None,
    test_beta=None,
    beam_size=None,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.encoder.acoustic_conditioning.load_state_dict(
        checkpoint["ac_state_dict"]
    )

    # Restore Stage-2 shared calibration gamma.
    if model.encoder.ac_location == "postconv_stage2":
        saved_calibration_logit = checkpoint.get(
            "calibration_logit",
            None,
        )
        if saved_calibration_logit is None:
            raise RuntimeError(
                "This postconv_stage2 model requires a checkpoint "
                "containing 'calibration_logit'."
            )
        model.encoder.calibration_logit.data.copy_(
            torch.as_tensor(
                saved_calibration_logit,
                device=model.encoder.calibration_logit.device,
                dtype=model.encoder.calibration_logit.dtype,
            )
        )

    # Restore adapted Whisper Conv1 for bandwise_frontend.
    if model.encoder.ac_location == "bandwise_frontend":
        saved_conv1_state = checkpoint.get(
            "conv1_state_dict",
            None,
        )

        if saved_conv1_state is None:
            raise RuntimeError(
                "This bandwise_frontend model requires a checkpoint "
                "containing 'conv1_state_dict'."
            )

        model.encoder.base_encoder.conv1.load_state_dict(
            saved_conv1_state
        )

    # Restore legacy bandwise_bridge lambda.
    if model.encoder.ac_location == "bandwise_bridge":
        saved_bridge_logit = checkpoint.get(
            "bridge_logit",
            None,
        )

        if saved_bridge_logit is None:
            raise RuntimeError(
                "This bandwise_bridge model requires a checkpoint "
                "containing 'bridge_logit'."
            )

        model.encoder.bridge_logit.data.copy_(
            torch.as_tensor(
                saved_bridge_logit,
                device=model.encoder.bridge_logit.device,
                dtype=model.encoder.bridge_logit.dtype,
            )
        )

    # Restore NEW bandwise_adapter state.
    if model.encoder.ac_location == "bandwise_adapter":
        saved_adapter_state = checkpoint.get(
            "adapter_state_dict",
            None,
        )
        saved_adapter_logit = checkpoint.get(
            "adapter_logit",
            None,
        )

        if saved_adapter_state is None:
            raise RuntimeError(
                "This bandwise_adapter model requires a checkpoint "
                "containing 'adapter_state_dict'."
            )

        if saved_adapter_logit is None:
            raise RuntimeError(
                "This bandwise_adapter model requires a checkpoint "
                "containing 'adapter_logit'."
            )

        model.encoder.ac_adapter.load_state_dict(
            saved_adapter_state
        )

        model.encoder.adapter_logit.data.copy_(
            torch.as_tensor(
                saved_adapter_logit,
                device=model.encoder.adapter_logit.device,
                dtype=model.encoder.adapter_logit.dtype,
            )
        )

    # Optional alpha override for old diagnostic experiments only.
    if test_alpha is not None:
        if model.encoder.ac_location == "postconv_stage2":
            raise ValueError(
                "--test_alpha is disabled for postconv_stage2."
            )
        for alpha_i in (
            model.encoder
            .acoustic_conditioning
            .alphas
        ):
            alpha_i.data.fill_(test_alpha)

    # test_beta belongs only to the old post-conv experiment.
    if model.encoder.ac_location != "postconv":
        if test_beta is not None:
            raise ValueError(
                "--test_beta is only supported for ac_location=postconv. "
                "It is not used by postconv_stage2 or bandwise modes."
            )
    else:
        if test_beta is not None:
            model.encoder.set_conditioning_beta(
                test_beta
            )
        else:
            model.encoder.set_conditioning_beta(
                1.0
            )

    model.eval()

    alpha_values = get_alpha_values(
        model
    )
    alpha = get_alpha_mean(
        model
    )

    print("=" * 80)
    print("Whisper + AC TEST")
    print("=" * 80)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Saved val  : {checkpoint.get('val_loss', 'N/A')}")
    print(f"AC location: {model.encoder.ac_location}")
    print(f"Mean alpha : {alpha:.6f}")
    print(f"Alpha list : {alpha_values}")

    if model.encoder.ac_location == "postconv_stage2":
        print(
            f"Calibration gamma: "
            f"{float(model.encoder.calibration_scale.detach().cpu()):.6f}"
        )
        print("Stage-1 AC      : FROZEN")
        print("Internal alphas : normalized to 1.0")
    elif model.encoder.ac_location == "bandwise_bridge":
        print(
            f"Bridge lambda: "
            f"{float(model.encoder.bridge_scale.detach().cpu()):.6f}"
        )
    elif model.encoder.ac_location == "bandwise_adapter":
        print(
            f"Adapter scale: "
            f"{float(model.encoder.adapter_scale.detach().cpu()):.6f}"
        )
    elif model.encoder.ac_location == "bandwise_frontend":
        print("Whisper Conv1: ADAPTED checkpoint weights loaded")
        print("Whisper Conv2: FROZEN pretrained weights")
    else:
        print(
            f"Test beta  : "
            f"{model.encoder.conditioning_beta:.6f}"
        )

    print("=" * 80)

    wav_map = read_key_value(wav_scp)
    acoustic_map = read_key_value(acoustic_scp)

    keys = [
        k for k in wav_map
        if k in acoustic_map
    ]

    print(
        f"[Test] wav={len(wav_map)} "
        f"acoustic={len(acoustic_map)} "
        f"matched={len(keys)}"
    )

    recog_dir = Path(output_dir) / "1best_recog"
    recog_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_text = recog_dir / "text"

    with open(
        output_text,
        "w",
        encoding="utf-8",
        buffering=1,
    ) as fout:

        for utt_id in tqdm(
            keys,
            desc=f"Decode alpha={alpha:.4f}",
        ):
            acoustic_feat = np.load(
                acoustic_map[utt_id]
            ).astype(np.float32)

            if acoustic_feat.shape != (9,):
                raise ValueError(
                    f"{utt_id}: expected acoustic feat (9,), "
                    f"got {acoustic_feat.shape}"
                )

            acoustic_feat = (
                torch.from_numpy(acoustic_feat)
                .unsqueeze(0)
                .to(device)
            )

            # Load once so the same waveform is used both for
            # determining the non-padding Mel region and decoding.
            audio = whisper.load_audio(
                wav_map[utt_id]
            )

            valid_mel_frames = torch.tensor(
                [get_valid_mel_frames(len(audio))],
                dtype=torch.long,
                device=device,
            )

            model.encoder.set_acoustic_feat(
                acoustic_feat,
                valid_mel_frames=valid_mel_frames,
            )

            try:
                decode_options = {
                    "language": "ko",
                    "task": "transcribe",
                    "verbose": False,
                    "fp16": False,
                    "temperature": 0.0,
                    "condition_on_previous_text": False,
                }

                if beam_size is not None:
                    decode_options["beam_size"] = beam_size

                result = model.transcribe(
                    audio,
                    **decode_options,
                )

            finally:
                model.encoder.clear_acoustic_feat()

            text_out = " ".join(
                result["text"].strip().split()
            )

            fout.write(
                f"{utt_id} {text_out}\n"
            )

    print(f"Saved: {output_text}")


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="small",
    )

    parser.add_argument(
        "--train_wav_scp",
        default="dump/raw/train/wav.scp",
    )

    parser.add_argument(
        "--train_text",
        default="dump/raw/train/text",
    )

    parser.add_argument(
        "--train_acoustic_scp",
        default=(
            "dump/hallu_acoustic_feats_fix/"
            "train/acoustic_feats.scp"
        ),
    )

    parser.add_argument(
        "--dev_wav_scp",
        default="dump/raw/dev/wav.scp",
    )

    parser.add_argument(
        "--dev_text",
        default="dump/raw/dev/text",
    )

    parser.add_argument(
        "--dev_acoustic_scp",
        default=(
            "dump/hallu_acoustic_feats_fix/"
            "dev/acoustic_feats.scp"
        ),
    )

    parser.add_argument(
        "--output_dir",
        default="exp/whisper_small_ac",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--grad_accum",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max_train",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max_dev",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
    )

    parser.add_argument(
        "--test_wav_scp",
        default="dump/raw/test_1k/wav.scp",
    )

    parser.add_argument(
        "--test_acoustic_scp",
        default=(
            "dump/hallu_acoustic_feats_fix/"
            "test_1k/acoustic_feats.scp"
        ),
    )

    parser.add_argument(
        "--test_output_dir",
        default=(
            "exp/whisper_small_multihead_ac/"
            "direct_gpu_test1k"
        ),
    )

    parser.add_argument(
        "--test_alpha",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--test_beta",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--resume_checkpoint",
        default=None,
    )

    parser.add_argument(
        "--start_epoch",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--beam_size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--ac_location",
        choices=[
            "postconv",
            "postconv_stage2",
            "bandwise_bridge",
            "bandwise_adapter",
            "bandwise_frontend",
            "bandwise_frozen",
        ],
        default="postconv",
    )

    parser.add_argument(
        "--stage1_checkpoint",
        default=None,
        help=(
            "Stage-1 post-conv checkpoint used to initialize "
            "postconv_stage2 training."
        ),
    )

    parser.add_argument(
        "--calibration_init",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--calibration_max",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--calibration_lr",
        type=float,
        default=1e-2,
    )

    parser.add_argument(
        "--fixed_alpha",
        type=float,
        default=None,
        help=(
            "Optional fixed per-band alpha. "
            "If omitted, alpha_i remain learnable."
        ),
    )

    parser.add_argument(
        "--frontend_lr",
        type=float,
        default=1e-5,
        help=(
            "Learning rate for Whisper Conv1 in bandwise_frontend. "
            "AC keeps using --lr."
        ),
    )

    parser.add_argument(
        "--bridge_init",
        type=float,
        default=0.1,
        help=(
            "Initial lambda for bandwise_bridge. "
            "The learned lambda is sigmoid-constrained to (0,1)."
        ),
    )

    parser.add_argument(
        "--adapter_hidden",
        type=int,
        default=128,
        help=(
            "Hidden channels in the grouped bandwise adapter. "
            "Must be divisible by 4."
        ),
    )

    parser.add_argument(
        "--adapter_scale_init",
        type=float,
        default=0.01,
        help=(
            "Initial latent residual scale for bandwise_adapter. "
            "Default 0.01."
        ),
    )

    parser.add_argument(
        "--adapter_scale_max",
        type=float,
        default=0.1,
        help=(
            "Maximum latent residual scale for bandwise_adapter. "
            "Actual scale = max * sigmoid(logit)."
        ),
    )

    args = parser.parse_args()

    seed_everything(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer = get_tokenizer(
        multilingual=True,
        language="ko",
        task="transcribe",
    )

    # ========================================================
    # Whisper
    # ========================================================

    print(
        f"Loading Whisper-{args.model}..."
    )

    model = whisper.load_model(
        args.model,
        device=device,
    )

    # Train stably with AMP
    model = model.float()

    base_encoder = model.encoder

    model.encoder = (
        AcousticConditionedWhisperEncoder(
            base_encoder=base_encoder,
            acoustic_dim=9,
            hidden_dim=128,
            dropout=0.1,
            ac_location=args.ac_location,
            bridge_init=args.bridge_init,
            adapter_hidden=args.adapter_hidden,
            adapter_scale_init=args.adapter_scale_init,
            adapter_scale_max=args.adapter_scale_max,
            calibration_init=args.calibration_init,
            calibration_max=args.calibration_max,
        )
        .to(device)
    )

    # ========================================================
    # Freeze ALL Whisper parameters
    # ========================================================

    for p in model.parameters():
        p.requires_grad_(False)

    # AC training policy
    if model.encoder.ac_location == "postconv_stage2":
        if args.fixed_alpha is not None:
            raise ValueError(
                "--fixed_alpha cannot be used with postconv_stage2."
            )

        if args.mode == "train" and args.resume_checkpoint is None:
            if args.stage1_checkpoint is None:
                raise ValueError(
                    "--stage1_checkpoint is required for fresh "
                    "postconv_stage2 training."
                )
            stage1_ckpt = torch.load(
                args.stage1_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            model.encoder.acoustic_conditioning.load_state_dict(
                stage1_ckpt["ac_state_dict"]
            )
            print(
                "Stage-1 alpha list  : "
                f"{get_alpha_values(model)}"
            )
            # Keep learned nets/gates, but remove head-specific scaling.
            for alpha_i in (
                model.encoder.acoustic_conditioning.alphas
            ):
                alpha_i.data.fill_(1.0)

        for p in model.encoder.acoustic_conditioning.parameters():
            p.requires_grad_(False)
        model.encoder.calibration_logit.requires_grad_(True)
    else:
        for p in (
            model.encoder
            .acoustic_conditioning
            .parameters()
        ):
            p.requires_grad_(True)

    # bandwise_bridge learns one global bounded bridge lambda.
    if model.encoder.bridge_logit is not None:
        model.encoder.bridge_logit.requires_grad_(True)

    # bandwise_adapter learns the grouped adapter and one bounded
    # latent residual scale. Whisper itself remains frozen.
    if model.encoder.ac_adapter is not None:
        for p in model.encoder.ac_adapter.parameters():
            p.requires_grad_(True)

    if model.encoder.adapter_logit is not None:
        model.encoder.adapter_logit.requires_grad_(True)

    # bandwise_frontend jointly adapts ONLY Whisper Conv1.
    # Conv2, Transformer encoder, and decoder remain frozen.
    if model.encoder.ac_location == "bandwise_frontend":
        for p in model.encoder.base_encoder.conv1.parameters():
            p.requires_grad_(True)

    # Optional fixed-alpha ablation.  By default alpha_i are learnable.
    if args.fixed_alpha is not None:
        for alpha_i in (
            model.encoder
            .acoustic_conditioning
            .alphas
        ):
            alpha_i.data.fill_(
                args.fixed_alpha
            )
            alpha_i.requires_grad_(False)

    n_total = sum(
        p.numel()
        for p in model.parameters()
    )

    n_trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("=" * 80)
    print("Whisper + Acoustic Conditioning")
    print("=" * 80)

    print(
        f"Model               : {args.model}"
    )

    print(
        f"Total params        : {n_total:,}"
    )

    print(
        f"Trainable params    : {n_trainable:,}"
    )

    if model.encoder.ac_location == "bandwise_frontend":
        print(
            "Whisper backbone    : PARTIAL (Conv1 trainable only)"
        )
    else:
        print(
            "Whisper backbone    : FROZEN"
        )

    print(
        "Whisper decoder     : FROZEN"
    )

    print(
        "AC acoustic_dim     : 9"
    )

    print(
        "AC input_dim        : "
        f"{model.encoder.acoustic_input_dim}"
    )

    print(
        "AC hidden_dim       : 128"
    )

    print(
        "AC location         : "
        f"{args.ac_location}"
    )

    if model.encoder.ac_location == "postconv_stage2":
        print(
            "AC alpha mode       : STAGE-1 FROZEN / internal alpha_i=1.0"
        )
        print(
            "Calibration gamma   : "
            f"init={args.calibration_init:.6f}, max={args.calibration_max:.6f}"
        )
        print(
            "Calibration LR      : "
            f"{args.calibration_lr:.2e}"
        )
        if args.stage1_checkpoint is not None:
            print(
                "Stage-1 checkpoint : "
                f"{args.stage1_checkpoint}"
            )
    else:
        print(
            "AC alpha mode       : "
            + (
                f"FIXED ({args.fixed_alpha})"
                if args.fixed_alpha is not None
                else "LEARNABLE"
            )
        )

    if model.encoder.ac_location in {
        "bandwise_bridge",
        "bandwise_adapter",
        "bandwise_frontend",
        "bandwise_frozen",
    }:
        print(
            "AC frequency bands  : "
            "0-19 / 20-39 / 40-59 / 60-79"
        )
        print(
            "AC valid-frame mask : ON (30-s padding excluded)"
        )
        print(
            "AC descriptor       : same 9-D vector -> all 4 heads"
        )

    if model.encoder.ac_location == "bandwise_bridge":
        print(
            "Bridge lambda init  : "
            f"{args.bridge_init:.6f}"
        )

    if model.encoder.ac_location == "bandwise_adapter":
        print(
            "Adapter hidden      : "
            f"{args.adapter_hidden}"
        )
        print(
            "Adapter grouping    : groups=4 (20 Mel bins/group)"
        )
        print(
            "Adapter scale init  : "
            f"{args.adapter_scale_init:.6f}"
        )
        print(
            "Adapter scale max   : "
            f"{args.adapter_scale_max:.6f}"
        )
        print(
            "Adapter final proj  : ZERO-INIT (starts as plain Whisper)"
        )

    if model.encoder.ac_location == "bandwise_frontend":
        print(
            "Whisper Conv1       : TRAINABLE"
        )
        print(
            "Whisper Conv1 LR    : "
            f"{args.frontend_lr:.2e}"
        )
        print(
            "Whisper Conv2       : FROZEN"
        )
        print(
            "Transformer/decoder : FROZEN"
        )

    if model.encoder.ac_location == "bandwise_frozen":
        print("Whisper Conv1       : FROZEN")
        print("Whisper Conv2       : FROZEN")
        print("Transformer/decoder : FROZEN")
        print("Trainable path      : AC ONLY")

    print("=" * 80)

    if args.mode == "test":

        if args.checkpoint is None:
            raise ValueError(
                "--checkpoint is required in test mode."
            )

        run_test(
            model=model,
            device=device,
            checkpoint_path=args.checkpoint,
            wav_scp=args.test_wav_scp,
            acoustic_scp=args.test_acoustic_scp,
            output_dir=args.test_output_dir,
            test_alpha=args.test_alpha,
            test_beta=args.test_beta,
            beam_size=args.beam_size,
        )

        return

    # ========================================================
    # Dataset
    # ========================================================

    train_set = WhisperACDataset(
        wav_scp=args.train_wav_scp,
        text_path=args.train_text,
        acoustic_scp=
            args.train_acoustic_scp,
        tokenizer=tokenizer,
        n_text_ctx=
            model.dims.n_text_ctx,
        max_items=args.max_train,
    )

    dev_set = WhisperACDataset(
        wav_scp=args.dev_wav_scp,
        text_path=args.dev_text,
        acoustic_scp=
            args.dev_acoustic_scp,
        tokenizer=tokenizer,
        n_text_ctx=
            model.dims.n_text_ctx,
        max_items=args.max_dev,
    )

    generator = torch.Generator()
    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_set,
        batch_size=
            args.batch_size,
        shuffle=True,
        num_workers=
            args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        generator=generator,
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=
            args.batch_size,
        shuffle=False,
        num_workers=
            args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    trainable_conditioning_params = [
        p
        for p in model.encoder.parameters()
        if p.requires_grad
    ]

    if model.encoder.ac_location == "postconv_stage2":
        optimizer = torch.optim.AdamW(
            [model.encoder.calibration_logit],
            lr=args.calibration_lr,
            weight_decay=0.0,
        )

    elif model.encoder.ac_location == "bandwise_frontend":
        ac_params = [
            p
            for p in (
                model.encoder
                .acoustic_conditioning
                .parameters()
            )
            if p.requires_grad
        ]

        conv1_params = [
            p
            for p in (
                model.encoder
                .base_encoder
                .conv1
                .parameters()
            )
            if p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": ac_params,
                    "lr": args.lr,
                },
                {
                    "params": conv1_params,
                    "lr": args.frontend_lr,
                },
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_conditioning_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    # ========================================================
    # Resume training
    # ========================================================

    resume_epoch = 0

    if args.resume_checkpoint is not None:

        checkpoint = torch.load(
            args.resume_checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        model.encoder.acoustic_conditioning.load_state_dict(
            checkpoint["ac_state_dict"]
        )

        if model.encoder.ac_location == "postconv_stage2":
            saved_calibration_logit = checkpoint.get(
                "calibration_logit",
                None,
            )
            if saved_calibration_logit is None:
                raise RuntimeError(
                    "Cannot resume postconv_stage2: checkpoint "
                    "does not contain 'calibration_logit'."
                )
            model.encoder.calibration_logit.data.copy_(
                torch.as_tensor(
                    saved_calibration_logit,
                    device=model.encoder.calibration_logit.device,
                    dtype=model.encoder.calibration_logit.dtype,
                )
            )

        if model.encoder.ac_location == "bandwise_frontend":
            saved_conv1_state = checkpoint.get(
                "conv1_state_dict",
                None,
            )
            if saved_conv1_state is None:
                raise RuntimeError(
                    "Cannot resume bandwise_frontend: checkpoint "
                    "does not contain 'conv1_state_dict'."
                )
            model.encoder.base_encoder.conv1.load_state_dict(
                saved_conv1_state
            )

        if model.encoder.bridge_logit is not None:
            saved_bridge_logit = checkpoint.get(
                "bridge_logit",
                None,
            )
            if saved_bridge_logit is None:
                raise RuntimeError(
                    "Cannot resume bandwise_bridge: checkpoint "
                    "does not contain 'bridge_logit'."
                )
            model.encoder.bridge_logit.data.copy_(
                torch.as_tensor(
                    saved_bridge_logit,
                    device=model.encoder.bridge_logit.device,
                    dtype=model.encoder.bridge_logit.dtype,
                )
            )

        if model.encoder.ac_location == "bandwise_adapter":
            saved_adapter_state = checkpoint.get(
                "adapter_state_dict",
                None,
            )
            saved_adapter_logit = checkpoint.get(
                "adapter_logit",
                None,
            )

            if saved_adapter_state is None:
                raise RuntimeError(
                    "Cannot resume bandwise_adapter: checkpoint "
                    "does not contain 'adapter_state_dict'."
                )

            if saved_adapter_logit is None:
                raise RuntimeError(
                    "Cannot resume bandwise_adapter: checkpoint "
                    "does not contain 'adapter_logit'."
                )

            model.encoder.ac_adapter.load_state_dict(
                saved_adapter_state
            )

            model.encoder.adapter_logit.data.copy_(
                torch.as_tensor(
                    saved_adapter_logit,
                    device=model.encoder.adapter_logit.device,
                    dtype=model.encoder.adapter_logit.dtype,
                )
            )

        resume_epoch = int(
            checkpoint.get("epoch", 0)
        )

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )
            print("Optimizer state    : RESTORED")
        else:
            print("Optimizer state    : NOT FOUND (fresh AdamW)")

        print("=" * 80)
        print("RESUME TRAINING")
        print(f"Checkpoint         : {args.resume_checkpoint}")
        print(f"Loaded epoch       : {resume_epoch}")
        print(
            f"Loaded val loss    : "
            f"{checkpoint.get('val_loss', 'N/A')}"
        )
        print("=" * 80)

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.resume_checkpoint is not None:
        best_val = float(
            checkpoint.get(
                "best_val",
                checkpoint.get("val_loss", float("inf")),
            )
        )
    else:
        best_val = float("inf")

    # ========================================================
    # Initial validation
    # ========================================================

    initial_val = evaluate(
        model,
        dev_loader,
        device,
    )

    print(
        f"Initial dev NLL/token: "
        f"{initial_val:.6f}"
    )

    # ========================================================
    # Training
    # ========================================================

    if args.start_epoch is not None:
        first_epoch = args.start_epoch
    else:
        first_epoch = resume_epoch + 1

    last_epoch = (
        first_epoch
        + args.epochs
        - 1
    )

    for epoch in range(
        first_epoch,
        last_epoch + 1,
    ):

        # Frozen Whisper always stays in inference mode.
        # Only AC enters train mode so its dropout remains active.
        model.eval()
        if model.encoder.ac_location != "postconv_stage2":
            model.encoder.acoustic_conditioning.train()
        if model.encoder.ac_adapter is not None:
            model.encoder.ac_adapter.train()
        if model.encoder.ac_location == "bandwise_frontend":
            model.encoder.base_encoder.conv1.train()
        optimizer.zero_grad(
            set_to_none=True
        )

        running = 0.0
        steps = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
        )

        for step, batch in enumerate(
            progress,
            start=1,
        ):

            mel = batch["mel"].to(
                device,
                non_blocking=True,
            )

            acoustic_feat = (
                batch["acoustic_feat"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            valid_mel_frames = (
                batch["valid_mel_frames"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            tokens_in = (
                batch["tokens_in"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            targets = (
                batch["targets"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            model.encoder.set_acoustic_feat(
                acoustic_feat,
                valid_mel_frames=valid_mel_frames,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(
                    device.type == "cuda"
                ),
            ):

                logits = model(
                    mel,
                    tokens_in,
                )

                loss = F.cross_entropy(
                    logits.reshape(
                        -1,
                        logits.size(-1),
                    ),
                    targets.reshape(-1),
                    ignore_index=-100,
                )

                loss = (
                    loss
                    / args.grad_accum
                )

            loss.backward()

            running += (
                loss.item()
                * args.grad_accum
            )

            steps += 1

            if (
                step % args.grad_accum == 0
                or step == len(train_loader)
            ):

                torch.nn.utils.clip_grad_norm_(
                    trainable_conditioning_params,
                    max_norm=1.0,
                )

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            alpha = get_alpha_mean(
                model
            )

            postfix = {
                "loss": f"{running/steps:.4f}",
                "alpha": f"{alpha:.4f}",
            }

            if model.encoder.calibration_scale is not None:
                postfix["gamma"] = (
                    f"{float(model.encoder.calibration_scale.detach().cpu()):.4f}"
                )

            if model.encoder.bridge_scale is not None:
                postfix["bridge"] = (
                    f"{float(model.encoder.bridge_scale.detach().cpu()):.4f}"
                )

            if model.encoder.adapter_scale is not None:
                postfix["adapter"] = (
                    f"{float(model.encoder.adapter_scale.detach().cpu()):.4f}"
                )

            progress.set_postfix(
                **postfix
            )

        model.encoder.clear_acoustic_feat()

        # ====================================================
        # Validation
        # ====================================================

        val_loss = evaluate(
            model,
            dev_loader,
            device,
        )

        alpha = get_alpha_mean(
            model
        )

        print()
        print("=" * 80)

        print(
            f"EPOCH {epoch}"
        )

        print(
            f"Train CE       : "
            f"{running/steps:.6f}"
        )

        print(
            f"Dev NLL/token  : "
            f"{val_loss:.6f}"
        )

        print(
            f"AC mean alpha  : "
            f"{alpha:.6f}"
        )

        print(
            f"AC alpha list  : "
            f"{get_alpha_values(model)}"
        )

        if model.encoder.ac_location == "postconv_stage2":
            print(
                f"Calibration gamma: "
                f"{float(model.encoder.calibration_scale.detach().cpu()):.6f}"
            )

        if model.encoder.ac_location == "bandwise_frontend":
            print(
                f"Conv1 LR       : "
                f"{args.frontend_lr:.2e}"
            )

        if model.encoder.bridge_scale is not None:
            print(
                f"Bridge lambda  : "
                f"{float(model.encoder.bridge_scale.detach().cpu()):.6f}"
            )

        if model.encoder.adapter_scale is not None:
            print(
                f"Adapter scale  : "
                f"{float(model.encoder.adapter_scale.detach().cpu()):.6f}"
            )

        print("=" * 80)

        checkpoint = {
            "epoch":
                epoch,

            "model_name":
                args.model,

            "ac_state_dict":
                model.encoder
                .acoustic_conditioning
                .state_dict(),

            "calibration_logit": (
                model.encoder.calibration_logit
                .detach()
                .cpu()
                if model.encoder.calibration_logit is not None
                else None
            ),

            "conv1_state_dict": (
                model.encoder
                .base_encoder
                .conv1
                .state_dict()
                if model.encoder.ac_location == "bandwise_frontend"
                else None
            ),

            "bridge_logit": (
                model.encoder.bridge_logit
                .detach()
                .cpu()
                if model.encoder.bridge_logit is not None
                else None
            ),

            "adapter_state_dict": (
                model.encoder.ac_adapter.state_dict()
                if model.encoder.ac_adapter is not None
                else None
            ),

            "adapter_logit": (
                model.encoder.adapter_logit
                .detach()
                .cpu()
                if model.encoder.adapter_logit is not None
                else None
            ),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "val_loss":
                val_loss,

            "best_val":
                min(best_val, val_loss),

            "args":
                vars(args),
        }

        torch.save(
            checkpoint,
            output_dir
            / "last.pt",
        )

        # Save every epoch checkpoint
        torch.save(
            checkpoint,
            output_dir / f"epoch_{epoch}.pt",
        )

        if val_loss < best_val:

            best_val = val_loss

            torch.save(
                checkpoint,
                output_dir
                / "best.pt",
            )

            print(
                f"NEW BEST DEV LOSS: "
                f"{best_val:.6f}"
            )

    print()
    print(
        f"Best dev loss: "
        f"{best_val:.6f}"
    )


if __name__ == "__main__":
    main()
