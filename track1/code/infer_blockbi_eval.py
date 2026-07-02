#!/usr/bin/env python3
"""Track-1 BLOCK-ONLINE USEF-TFGridNet EVAL inference -- block-bi NO-FINETUNE init.

Adapted from /data/home/kts123/realtse/usef_tse/infer_causal_eval.py. The ONLY
differences from the forward-only causal floor inference are the model build +
weight load:

  * model is built from usef_lookahead/models (Tar_Model with inter_block=24:
    bidirectional inter_rnn run over non-overlapping 24-frame chunks), SAME
    STFT/dims as make_blockbi_init.py.
  * weights are loaded DIRECTLY (strict) from
    blockbi_init_ep38_W24.pth.tar's model_state_dict -- which is the FULL
    bidirectional ep38 warm-start (100% preserved, 0 reinit). We assert 0
    missing / 0 unexpected so we cannot silently run a wrong checkpoint.

EVERYTHING ELSE is identical to infer_causal_eval.py's EVAL1 path: SPLITS,
load_pairs/load_mapping, the <corpus>/wav/<mix>-<enr>.wav layout, per-corpus
tse_audio_mapping.csv, the --finalize rebuild, resume-skip, NaN guard.

Usage:
    CUDA_VISIBLE_DEVICES=5 \
      /data/home/kts123/miniconda3/envs/usef/bin/python infer_blockbi_eval.py \
        --split EVAL1 \
        --out /data/home/kts123/realtse/blockbi_ep38_W24_eval/EVAL1/usef_blockbi_ep38_W24
    # then finalize:
    ... infer_blockbi_eval.py --split EVAL1 --out <same> --finalize
"""
import os, sys, csv, argparse, time, glob
import numpy as np, torch, librosa, soundfile as sf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Build model from OUR block-online models/ + utils/ (NOT usef_tse / usef_causal_ft).
sys.path.insert(0, THIS_DIR)

MODEL_SR = 16000
W_DEFAULT = 24
INIT_CKPT_DEFAULT = os.path.join(THIS_DIR, "blockbi_init_ep38_W24.pth.tar")

# ---- splits (verbatim from infer_causal_eval.py) -------------------------
SPLITS = {
    "EVAL1": dict(
        meta_dir="/data/home/kts123/realtse/REAL-T-eval1/EVAL1",
        mapping="/data/home/kts123/realtse/REAL-T-eval1/mapping.csv",
        corpora=["AliMeeting", "AMI", "CHiME6", "DipCo"],
    ),
    "EVAL2": dict(
        meta_dir="/data/home/kts123/realtse/REAL-T-eval2/EVAL2",
        mapping="/data/home/kts123/realtse/REAL-T-eval2/mapping.csv",
        corpora=["unseen_CN", "unseen_EN"],
    ),
    "DEV": dict(
        meta_dir="/data/home/kts123/realtse/REAL-T-dev/DEV",
        mapping="/data/home/kts123/realtse/REAL-T-dev/mapping.csv",
        corpora=["AliMeeting", "AISHELL-4", "AMI", "DipCo", "CHiME6"],
    ),
}


def build_blockbi_model(W):
    """Build the block-online Tar_Model: SAME arch / STFT / dims as
    make_blockbi_init.py (all5 causal stage + inter_block=W)."""
    from utils.feature import STFT, iSTFT
    from models.local.TFgridnet import TF_gridnet_attentionblock
    from models.model_USEF_TFGridNet import Tar_Model

    stft = STFT(n_fft=128, hop_length=64, win_length=128)
    istft = iSTFT(n_fft=128, hop_length=64, win_length=128)
    real_att = TF_gridnet_attentionblock(
        emb_dim=128, n_freqs=65, n_head=4, approx_qk_dim=512
    )
    model = Tar_Model(
        stft=stft, istft=istft, real_att=real_att,
        n_freqs=65, hidden_channels=256, n_head=4,
        emb_dim=128, emb_ks=1, emb_hs=1, num_layers=6,
        p_rnn=True, p_attn=True, p_conv=True, p_norm=True, p_inorm=True,
        inter_block=W,
    )
    return model


def load_init_strict(model, ckpt_path):
    """STRICT load of a block-bi checkpoint's model_state_dict. Assert 0 missing
    / 0 unexpected -- refuse to run a mismatched checkpoint.

    Handles BOTH layouts:
      * my make_blockbi_init.py outputs (un-prefixed keys), and
      * trainer checkpoints saved from a DataParallel-wrapped model, whose keys
        all carry a leading "module." prefix. We strip "module." (mirroring
        infer_realt_usef_best.py / forward_half_load) BEFORE the strict load so
        param names line up; strict=True then still guarantees an exact,
        complete match (no silent partial loads)."""
    info = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = info["model_state_dict"] if "model_state_dict" in info else info
    from collections import OrderedDict
    n_stripped = sum(1 for k in sd if k.startswith("module."))
    if n_stripped:
        sd = OrderedDict(
            (k[len("module."):] if k.startswith("module.") else k, v)
            for k, v in sd.items()
        )
        print(f"[load] stripped 'module.' prefix from {n_stripped} keys "
              f"(DataParallel checkpoint)", flush=True)
    res = model.load_state_dict(sd, strict=True)
    # strict=True raises on any mismatch; this assert is belt-and-suspenders.
    assert len(getattr(res, "missing_keys", [])) == 0, f"missing keys: {res.missing_keys}"
    assert len(getattr(res, "unexpected_keys", [])) == 0, f"unexpected keys: {res.unexpected_keys}"
    return sd


# ---- I/O helpers (verbatim from infer_causal_eval.py) --------------------
def load_mapping(p):
    with open(p) as f:
        return {r["utterance"]: r["path"] for r in csv.DictReader(f)}


def load_pairs(meta):
    seen, pairs = set(), []
    with open(meta) as f:
        for r in csv.DictReader(f):
            k = (r["mixture_utterance"], r["enrolment_speakers_utterance"])
            if k not in seen:
                seen.add(k)
                pairs.append(k)
    return pairs


def finalize_mapping(out_dir, corpora):
    grand = 0
    for corpus in corpora:
        wav_dir = os.path.join(out_dir, corpus, "wav")
        if not os.path.isdir(wav_dir):
            print(f"[finalize][skip] no wav dir {wav_dir}", flush=True)
            continue
        wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
        rows = [(os.path.basename(w)[:-4], os.path.abspath(w)) for w in wavs]
        with open(os.path.join(out_dir, corpus, "tse_audio_mapping.csv"),
                  "w", newline="\n") as fo:
            w = csv.writer(fo, lineterminator="\n")
            w.writerow(["utterance", "path"])
            w.writerows(rows)
        grand += len(rows)
        print(f"[finalize][{corpus}] mapping rows={len(rows)}", flush=True)
    print(f"[finalize][DONE] {out_dir} total={grand}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(SPLITS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=W_DEFAULT, help="inter_block W")
    ap.add_argument("--init", default=INIT_CKPT_DEFAULT, help="block-bi init ckpt")
    ap.add_argument("--corpus", default="", help="comma-separated corpus subset")
    ap.add_argument("--finalize", action="store_true",
                    help="rebuild canonical tse_audio_mapping.csv per corpus from wavs on disk")
    a = ap.parse_args()
    cfg = SPLITS[a.split]

    if a.finalize:
        finalize_mapping(a.out, cfg["corpora"])
        return

    print(f"[build] block-online model inter_block={a.block}", flush=True)
    model = build_blockbi_model(a.block)
    # sanity: confirm block-bi structure BEFORE loading/running.
    n_bi = sum(int(b.inter_rnn.bidirectional) for b in model.dual_mdl)
    n_blk = len(model.dual_mdl)
    il_in = model.dual_mdl[0].inter_linear.in_features
    blk_set = [int(b.inter_block) if b.inter_block is not None else None for b in model.dual_mdl]
    print(f"[build] inter_rnn bidirectional {n_bi}/{n_blk} | inter_linear.in={il_in} "
          f"(expect 512) | inter_block per layer={blk_set}", flush=True)
    assert n_bi == n_blk, "block-bi model NOT bidirectional -- abort"
    assert il_in == 512, "inter_linear not 512-wide -- not the bidirectional arch -- abort"
    assert all(b == a.block for b in blk_set), "inter_block not set on all blocks -- abort"

    print(f"[load] strict-loading init: {a.init}", flush=True)
    assert os.path.isfile(a.init), f"init ckpt not found: {a.init}"
    load_init_strict(model, a.init)
    print(f"[load] STRICT OK: 0 missing / 0 unexpected", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    mapping = load_mapping(cfg["mapping"])
    print(f"[start] split={a.split} block={a.block} out={a.out} dev={dev} "
          f"cuda_dev={os.environ.get('CUDA_VISIBLE_DEVICES','')}", flush=True)

    grand, t_all = 0, time.time()
    corpora = a.corpus.split(",") if a.corpus else cfg["corpora"]
    for corpus in corpora:
        meta = os.path.join(cfg["meta_dir"], f"{corpus}_meta.csv")
        if not os.path.exists(meta):
            print(f"[skip] {meta}", flush=True)
            continue
        pairs = load_pairs(meta)
        wav_dir = os.path.join(a.out, corpus, "wav")
        os.makedirs(wav_dir, exist_ok=True)
        rows, nan_cnt, t0 = [], 0, time.time()
        with torch.no_grad():
            for i, (mix, enr) in enumerate(pairs):
                mp, ep = mapping.get(mix), mapping.get(enr)
                if mp is None or ep is None:
                    continue
                utt = f"{mix}-{enr}"
                dst = os.path.join(wav_dir, f"{utt}.wav")
                if os.path.exists(dst):
                    rows.append((utt, dst))
                    continue
                mw, _ = librosa.load(mp, sr=MODEL_SR)
                ew, _ = librosa.load(ep, sr=MODEL_SR)
                mt = torch.from_numpy(mw).unsqueeze(0).to(dev)
                et = torch.from_numpy(ew).unsqueeze(0).to(dev)
                est = model(mt, et).squeeze().cpu().numpy().astype(np.float32)
                if np.isnan(est).any():
                    nan_cnt += 1
                    est = np.nan_to_num(est)
                sf.write(dst, est, MODEL_SR, subtype="PCM_16")
                rows.append((utt, dst))
                if (i + 1) % 100 == 0:
                    print(f"  [{corpus}] {i+1}/{len(pairs)} ({time.time()-t0:.0f}s)", flush=True)
                if dev == "cuda":
                    torch.cuda.empty_cache()
        with open(os.path.join(a.out, corpus, "tse_audio_mapping.csv"),
                  "w", newline="\n") as fo:
            w = csv.writer(fo, lineterminator="\n")
            w.writerow(["utterance", "path"])
            w.writerows(rows)
        grand += len(rows)
        print(f"[{corpus}] wrote={len(rows)} nan={nan_cnt} in {time.time()-t0:.0f}s", flush=True)
    print(f"[DONE] {a.split} block={a.block} total={grand} in {time.time()-t_all:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
