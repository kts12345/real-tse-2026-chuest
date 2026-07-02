#!/usr/bin/env python3
"""RE-USE (nvidia/RE-USE SEMamba SE) inference on crown EVAL wavs -> 16k mono PCM16 wav.
Exact inference.py logic: model(mag,pha)->amp_g; expm1(relu) ONLY for zero-portion mask; RAW amp_g
(zeroed cols) -> mag_phase_istft(+compress_factor decompresses). Run from reuse_model dir, meanflow env (mamba_ssm)."""
import os, sys, glob, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/data/realtse_step1/reuse_model')
import torch, torchaudio, soundfile as sf, numpy as np
import torch.nn as nn
from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator_SEMamba_time_d4 import SEMamba
from utils.util import load_config, pad_or_trim_to_match
RELU = nn.ReLU()
cfg = load_config('/data/realtse_step1/reuse_model/config.json')
s = cfg['stft_cfg']; CF = cfg['model_cfg']['compress_factor']; BASE_SR = s['sampling_rate']
m = SEMamba.from_pretrained('nvidia/RE-USE', cfg=cfg).cuda().eval()
def even(v): v = int(round(v)); return v if v % 2 == 0 else v + 1
CROWN = '/data/realtse_step1/ep16_bestof_stage'
OUT = '/data/realtse_step1/reuse_out'
CORP = {'EVAL1': ['AMI', 'CHiME6', 'AliMeeting', 'DipCo'], 'EVAL2': ['unseen_CN', 'unseen_EN']}
@torch.no_grad()
def reuse(w, sr):
    nf, hp, wn = even(s['n_fft'] * sr // BASE_SR), even(s['hop_size'] * sr // BASE_SR), even(s['win_size'] * sr // BASE_SR)
    mag, pha, _ = mag_phase_stft(w, n_fft=nf, hop_size=hp, win_size=wn, compress_factor=CF, center=True, addeps=False)
    amp_g, pha_g, _ = m(mag, pha)
    mag_d = torch.expm1(RELU(amp_g))
    zp = torch.sum(mag_d == 0, 1) / mag_d.shape[1]
    amp_g[:, :, (zp > 0.5)[0]] = 0
    out = mag_phase_istft(amp_g, pha_g, nf, hp, wn, CF)
    return pad_or_trim_to_match(w.detach(), out, pad_value=1e-8)
done = 0; bad = 0
for sp, cs in CORP.items():
    for c in cs:
        wd = f'{CROWN}/{sp}/ep16_enh_6way/{c}/wav'
        if not os.path.isdir(wd): print(f'SKIP {sp}/{c} ({wd})', flush=True); continue
        od = f'{OUT}/{sp}/{c}/wav'; os.makedirs(od, exist_ok=True)
        for wf in sorted(glob.glob(wd + '/*.wav')):
            w, sr = torchaudio.load(wf)
            if w.shape[0] > 1: w = w.mean(0, keepdim=True)
            try:
                out = reuse(w.cuda(), sr).squeeze(0).cpu().numpy()
            except Exception as e:
                print(f'FAIL {os.path.basename(wf)}: {e}', flush=True); bad += 1; continue
            if sr != 16000:
                import librosa; out = librosa.resample(out, orig_sr=sr, target_sr=16000)
            out = np.clip(out, -1.0, 1.0)
            sf.write(f'{od}/{os.path.basename(wf)}', out.astype(np.float32), 16000, subtype='PCM_16')
            done += 1
        print(f'[{sp}/{c}] reuse {len(glob.glob(od+"/*.wav"))}', flush=True)
print(f'REUSE_INFER_DONE done={done} bad={bad}', flush=True)
