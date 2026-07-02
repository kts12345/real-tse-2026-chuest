#!/usr/bin/env python3
"""USEF-TSE epoch13 REAL-T inference -> official scorer-input layout.

For each split (DEV/EVAL1/EVAL2) and each corpus meta CSV, run TSE on EVERY
(mixture_utterance, enrolment_speakers_utterance) pair (the scorer joins on
utterance = mixture-enrol, one output per pair) and emit:
    <OUT>/<corpus>/wav/<utterance>.wav
    <OUT>/<corpus>/tse_audio_mapping.csv   (columns: utterance,path)

Run one split per invocation via --split.
"""
import os, sys, csv, argparse, time
import numpy as np
import torch
import librosa
import soundfile as sf

USEF_CODE = "/data/generative_tse/USEF-TSE-code"
sys.path.insert(0, USEF_CODE)
from hyperpyyaml import load_hyperpyyaml
from collections import OrderedDict

CKPT = os.environ.get("USEF_CKPT",
       "/data/generative_tse/USEF-TSE-code/chkpt/chkpt/usef_realt_16k/epoch16.pth.tar")
STAGE1_CKPT = os.environ.get("USEF_STAGE1_CKPT", "/data/generative_tse/USEF-TSE-code/chkpt/chkpt/track2_crown_curriculum/epoch15.pth.tar")
OUT_TAG = os.environ.get("USEF_OUT_TAG", "usef_best")  # override per-ckpt so extractions don't clobber
CONFIG = os.environ.get("USEF_CONFIG",  # portable: PC24 sets this to the bundled config path
       "/data/generative_tse/USEF-TSE-weights/chkpt/USEF-TFGridNet/config.yaml")
MODEL_SR = 16000

SPLITS = {
    "DEV": {
        "meta_dir": "/data/realtse_step1/dev_data/REAL-T-dev/DEV",
        "mapping": "/data/realtse_step1/dev_data/REAL-T-dev/mapping.csv",
        "out": "/data/generative_tse/usef_score/output_REAL-T-dev/DEV/usef_best",
        "corpora": ["AliMeeting", "AISHELL-4", "AMI", "DipCo", "CHiME6"],
    },
    "EVAL1": {
        "meta_dir": "/data/realtse_step1/eval_data/REAL-T-eval1/EVAL1",
        "mapping": "/data/realtse_step1/eval_data/REAL-T-eval1/mapping.csv",
        "out": "/data/generative_tse/usef_score/output_REAL-T-eval/EVAL1/usef_best",
        "corpora": ["AliMeeting", "AMI", "CHiME6", "DipCo"],
    },
    "EVAL2": {
        "meta_dir": "/data/realtse_step1/eval_data/REAL-T-eval2/EVAL2",
        "mapping": "/data/realtse_step1/eval_data/REAL-T-eval2/mapping.csv",
        "out": "/data/generative_tse/usef_score/output_REAL-T-eval/EVAL2/usef_best",
        "corpora": ["unseen_CN", "unseen_EN"],
    },
}


def load_mapping(path):
    m = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            m[row["utterance"]] = row["path"]
    return m


def load_pairs(meta_path):
    """All (mix, enr) rows in file order, de-duplicated on the (mix,enr) key."""
    seen = set()
    pairs = []
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            k = (row["mixture_utterance"], row["enrolment_speakers_utterance"])
            if k in seen:
                continue
            seen.add(k)
            pairs.append(k)
    return pairs


def build_model():
    with open(CONFIG) as f:
        config = load_hyperpyyaml(f.read())
    return config["modules"]["masknet"]


def load_ckpt(model, ckpt_path=None):
    info = torch.load(ckpt_path or CKPT, map_location="cpu")
    raw = info.get("model_state_dict", info.get("state_dict", info))
    state = OrderedDict()
    for k, v in raw.items():
        name = k.replace("module.", "")
        if "convolution_module." not in name:  # legacy ckpts need remap; new-format ckpts already have it (avoid greedy double-transform -> 8 phantom unexpected keys)
            name = name.replace("convolution_", "convolution_module.")
        state[name] = v
    res = model.load_state_dict(state, strict=False)
    return len(res.missing_keys), len(res.unexpected_keys), list(res.unexpected_keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(SPLITS))
    ap.add_argument("--n", type=int, default=0, help="per-corpus cap (0 = all)")
    ap.add_argument("--shard", default="", help="i/N multi-GPU split e.g. 0/8 (pairs[i::N]); wavs share dir (no collision), mapping is shard-tagged -> merge after")
    args = ap.parse_args()
    cfg = dict(SPLITS[args.split])
    cfg["out"] = cfg["out"].replace("usef_best", OUT_TAG)  # per-ckpt out dir (USEF_OUT_TAG)

    mapping = load_mapping(cfg["mapping"])
    model = build_model()
    miss, unexp, unexp_keys = load_ckpt(model)
    print(f"[ckpt] {os.path.basename(CKPT)} -> out={cfg['out']} missing={miss} unexpected={unexp}", flush=True)
    if unexp:
        print(f"[ckpt] ignoring {unexp} non-masknet keys (training-only aux e.g. PVAD head, not used in extraction): {unexp_keys}", flush=True)
    assert miss == 0, "CKPT MASKNET INCOMPLETE (missing>0) -> abort (silent-fail guard)"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()

    # ---- Stage-2 cascade: load FROZEN Stage1 (ep15) ----
    stage1 = build_model()
    s1_miss, s1_unexp, _s1k = load_ckpt(stage1, STAGE1_CKPT)
    print(f"[stage1] {os.path.basename(STAGE1_CKPT)} missing={s1_miss} unexpected={s1_unexp}", flush=True)
    assert s1_miss == 0, "STAGE1 CKPT INCOMPLETE (missing>0) -> abort (silent-fail guard)"
    stage1.to(dev).eval()

    grand = 0
    t_all = time.time()
    for corpus in cfg["corpora"]:
        meta = os.path.join(cfg["meta_dir"], f"{corpus}_meta.csv")
        if not os.path.exists(meta):
            print(f"[skip] no meta {meta}", flush=True)
            continue
        pairs = load_pairs(meta)
        if args.n:
            pairs = pairs[: args.n]
        if args.shard:
            _si, _sn = (int(x) for x in args.shard.split("/"))
            pairs = pairs[_si::_sn]   # even 1/N slice of this corpus for multi-GPU split
        wav_dir = os.path.join(cfg["out"], corpus, "wav")
        os.makedirs(wav_dir, exist_ok=True)
        rows, skipped, silent, nan_cnt = [], [], 0, 0
        t0 = time.time()
        with torch.no_grad():
            for i, (mix, enr) in enumerate(pairs):
                mp, ep = mapping.get(mix), mapping.get(enr)
                if mp is None or ep is None:
                    skipped.append((mix, enr))
                    continue
                utt = f"{mix}-{enr}"
                dst = os.path.join(wav_dir, f"{utt}.wav")
                if os.path.exists(dst):       # resume: skip already-inferred wav
                    rows.append((utt, dst))
                    continue
                mw, _ = librosa.load(mp, sr=MODEL_SR)
                ew, _ = librosa.load(ep, sr=MODEL_SR)
                mt = torch.from_numpy(mw).unsqueeze(0).to(dev)
                et = torch.from_numpy(ew).unsqueeze(0).to(dev)
                # ---- Stage-2 cascade: Stage1(frozen) -> refiner -> structural mask blend (학습과 동일) ----
                _s1 = stage1(mt, et)
                s1_wav = _s1["est_source"] if isinstance(_s1, dict) else _s1
                s1_pvad = _s1["pvad_logit"]
                _r = model(s1_wav, et)
                r_est = _r["est_source"] if isinstance(_r, dict) else _r
                _vad = torch.sigmoid(s1_pvad)
                _vad = (_vad >= 0.5).float()                                  # struct_mask_thresh (config 0.5)
                _hop = max(1, round(s1_wav.shape[-1] / max(1, s1_pvad.shape[-1])))  # ~STFT hop
                _vw = _vad.repeat_interleave(_hop, dim=-1)
                _Lm = min(r_est.shape[-1], s1_wav.shape[-1], _vw.shape[-1])
                est_t = _vw[..., :_Lm] * r_est[..., :_Lm] + (1.0 - _vw[..., :_Lm]) * s1_wav[..., :_Lm]
                est = est_t.squeeze().cpu().numpy().astype(np.float32)
                if np.isnan(est).any():
                    nan_cnt += 1
                    est = np.nan_to_num(est)
                utt = f"{mix}-{enr}"
                dst = os.path.join(wav_dir, f"{utt}.wav")
                sf.write(dst, est, MODEL_SR)
                # silence-gate dump (PC24 lever-1): per-frame sigmoid(Stage1 PVAD) float16.
                # Stage1=ep15 FIXED across refiner epochs -> gate identical; PC24 upsample+thresh sweep.
                _gd = os.path.join(cfg["out"], corpus, "gate")
                os.makedirs(_gd, exist_ok=True)
                np.save(os.path.join(_gd, f"{utt}.npy"),
                        torch.sigmoid(s1_pvad).squeeze().detach().cpu().numpy().astype(np.float16))
                rms = float(np.sqrt(np.mean(est ** 2))) if est.size else 0.0
                if rms < 1e-4:
                    silent += 1
                rows.append((utt, dst))
                if (i + 1) % 50 == 0:
                    print(f"  [{corpus}] {i+1}/{len(pairs)} ({time.time()-t0:.0f}s)", flush=True)
                if dev == "cuda":
                    torch.cuda.empty_cache()
        _mtag = ("_sh" + args.shard.replace("/", "of")) if args.shard else ""
        mapping_csv = os.path.join(cfg["out"], corpus, f"tse_audio_mapping{_mtag}.csv")
        with open(mapping_csv, "w", newline="\n") as fo:
            w = csv.writer(fo, lineterminator="\n")
            w.writerow(["utterance", "path"])
            w.writerows(rows)
        grand += len(rows)
        print(f"[{corpus}] wrote={len(rows)} skipped={len(skipped)} "
              f"near-silent={silent} nan={nan_cnt} in {time.time()-t0:.0f}s -> {mapping_csv}", flush=True)
        if skipped:
            print(f"   sample skipped: {skipped[:3]}", flush=True)

    print(f"\n[{args.split} DONE] total wavs={grand} in {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
