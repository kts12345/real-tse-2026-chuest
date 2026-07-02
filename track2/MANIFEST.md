# Track 2 (#188 submit_reuse_5) reproduction files

Contents mirror the system described in the system description paper, sections 3.1-3.7.

## weights/
- `epoch16.pth.tar` (188 MB), md5 `3c330b7ec87b8677a926f2fd7be1581b`: stage-1 USEF-TFGridNet checkpoint ("crown base"). Matches the md5 stated in the paper.

## scripts/
- `infer_realt_stage2_cascade.py`, md5 `a69548578dfcb8a88711aaf6fa3b82fe`: stage-1 inference driver. runs TSE on every (mixture, enrollment) pair of DEV/EVAL1/EVAL2 and emits the official scorer input layout (`<corpus>/wav/<utterance>.wav` + `tse_audio_mapping.csv`).
- `reuse_infer.py`, md5 `71304044b2e7f6c502a2a1763e3cbcde`: stage-2. applies the released nvidia/RE-USE enhancer to the stage-1 outputs (16 kHz mono PCM16). The RE-USE weights themselves are NOT included (license: research and development only); download from Hugging Face `nvidia/RE-USE`.
- `reuse_route_corpus.py`, md5 `90931934be448d25d001a4eae7d0fef1`: per-file eps-0.05 routing between the stage-1 output and the enhanced output, using WeSpeaker embeddings via `wespeakerruntime`. Writes the per-corpus routing CSVs and the routed submission wavs.
- `usef_tfgridnet_config.yaml`, md5 `293a7aeeba43eda9f4cda4678ba33f37`: stage-1 model configuration (STFT 128/64, hidden 256, emb 128, 4 heads, 6 layers).

## routing_csv/
22 files, `reuse_route_EVAL1_*.csv` + `reuse_route_EVAL2_*.csv` (columns: split, corpus, utterance, sim_raw, sim_enh, chosen). Aggregate: 3,210 crown / 1,790 reuse over the 5,000 evaluation utterances; decision boundary exactly 0.0500. Concatenated md5 fingerprint `62d1cfa6`.

## Reproduction order
1. Stage-1 extraction: `infer_realt_stage2_cascade.py` with `epoch16.pth.tar` and `usef_tfgridnet_config.yaml` (standard USEF-TSE runtime).
2. Stage-2 enhancement: `reuse_infer.py` with the `nvidia/RE-USE` release.
3. Routing: `reuse_route_corpus.py` (eps 0.05), which reproduces the routing CSVs and the final submitted waveforms.

Not included by policy: RE-USE weights (redistribution not permitted by its research-and-development-only license), and any DEV/EVAL audio (challenge data). Paths inside the scripts refer to the original compute node layout and need adjusting to the local environment.
