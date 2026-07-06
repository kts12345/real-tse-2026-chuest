# Track 1 #279 — Dataset & Pretrained-Model Declaration
*For the official System Description ("data usage" + every pretrained model must be reported with citations/links). Scope: #279 = `usef_b5_dnsmos17s` ONLY.*

**EN:** Submission #279 was trained only on open-source corpora (training splits) plus standard augmentation. The REAL-T / REAL-TSE challenge DEV and EVAL audio was used solely for validation/testing and was **never used as training data** — only utterance-length metadata informed length-matched cropping. All speaker-embedding and DNSMOS components are **training-time losses** and are not part of the deployed network (they do not affect inference, latency, or the submitted weights).

**KR:** 제출 #279는 오픈소스 코퍼스(학습 split)와 표준 증강만으로 학습했다. REAL-T/REAL-TSE 챌린지 DEV·EVAL 오디오는 검증/테스트에만 사용했고 **학습 데이터로는 전혀 쓰지 않았다**(발화 길이 메타데이터만 length-matching에 활용). 화자임베딩·DNSMOS 구성요소는 모두 **학습시점 손실**이며 배포 네트워크에 포함되지 않는다(추론·지연·제출 가중치에 무영향).

## Warm-start lineage (#279)
Public USEF-TSE (USEF-TFGridNet, WHAM! variant) → 16 kHz adaptation fine-tune (AliMeeting training-partition data; internal run name `usef_realt_16k`, where "realt" is a project prefix, not REAL-T corpus audio) → F1 fine-tune (v3combo, ep71) → bidirectional→block-online conversion (`blockbi_init_ep71_W24`, 100% verbatim weight transfer) → b5 (Libri2Mix train-clean-100, 4 s) → **#279 `usef_b5_dnsmos17s`** (campaign_v2_matchlen 17 s + differentiable DNSMOS surrogate + ECAPA + energy losses).

## A. Datasets
| # | Dataset (role) | Source & citation | License | Split |
|---|---|---|---|---|
| A1 | **campaign_v2_matchlen** — PRIMARY #279 fine-tune data (17 s, EVAL-length-matched, re-synthesized mixtures: real RIR RT60 0.5–0.8 s, SIR U(−5,+5) dB, 3–4 spk, target-silent 45–58%) | derivative re-mix of A1a–d | derivative (A1a–d apply) | — |
| A1a | ↳ AliMeeting (ZH, 40%) | M2MeT, Yu et al. ICASSP 2022 | **CC BY-SA 4.0** (OpenSLR SLR119) | **train** |
| A1b | ↳ CHiME-6 (EN, 27%) | Watanabe et al. 2020 | research via challenge | **train** (S03) |
| A1c | ↳ AMI Meeting Corpus (EN, 23%) | Carletta et al. 2006 | CC BY 4.0 | **train** (ami_train) |
| A1d | ↳ AISHELL-4 (ZH, 10%) | Fu et al. Interspeech 2021 | **CC BY-SA 4.0** (OpenSLR SLR111) | **train** (767/7600 = 10.1%) |
| A2 | MUSAN — additive noise (train-time aug) | Snyder et al. 2015, arXiv:1510.08484 | CC BY 4.0 | noise+music only (speech excluded) |
| A3 | RIRS_NOISES (simulated_rirs) — reverb aug | Ko et al. 2017, OpenSLR SLR28 | Apache-2.0 | — |
| A4 | Libri2Mix train-clean-100 — b5-stage fine-tune | Cosentino et al. 2020, arXiv:2005.11262 (over LibriSpeech) | CC BY 4.0 | train-clean-100 |
| A5 | WHAM! — base-checkpoint pretraining (via B1) | Wichern et al. Interspeech 2019 (WSJ0-2mix + WHAM noise) | WHAM! CC BY-NC 4.0; WSJ0 LDC | — |
| A6 | 16k-adaptation set (simulated near-field mixtures + AliMeeting far-field training recordings) and v3combo (493 speakers / 6,000 mixtures) — our derived fine-tune sets; "realt" in internal names is a project prefix, no REAL-T corpus audio | this work, derived from AliMeeting / AMI training splits | inherits source licenses | **train** |
| A7 | REAL-T / REAL-TSE DEV (1991) / EVAL (5000) | organizer-provided real conversational data | challenge | **VALIDATION/TEST ONLY — never trained on (length metadata only)** |

## B. Pretrained models
| # | Model (role in #279) | Identity / source & citation | License |
|---|---|---|---|
| B1 | **USEF-TSE / USEF-TFGridNet** — warm-start base of the whole chain = the public **8 kHz USEF-TFGridNet release (WHAM! variant)**, fine-tuned to 16 kHz on AliMeeting training-partition data in our chain (exact variant fixed by our configs) | HF `ZBang/USEF-TSE`; USEF-TSE, Zeng et al., **arXiv:2409.02615** | **CC BY-NC 4.0** (verified via repo README) |
| B2 | **ECAPA-TDNN** `speechbrain/spkrec-ecapa-voxceleb` — frozen speaker-cosine loss (train-time only, not saved) | Desplanques et al. 2020; SpeechBrain, Ravanelli et al. 2021 | Apache-2.0 |
| B3 | **Microsoft DNSMOS** `sig_bak_ovr.onnx` (P.835) — differentiable surrogate loss (train-time only, not saved) | Reddy et al. 2022 "DNSMOS P.835" (ICASSP), DNS-Challenge; md5 `f79cd6a01fe88edc124df774008a926e` | DNS-Challenge / Microsoft (research) |

**Not used by #279** (declared for transparency): WeSpeaker ResNet34-LM and Silero-VAD are present in the loss module but their weights are 0 in #279's config; GAN/WavLM/over-attenuation weights are all 0 → no associated pretrained models used.
