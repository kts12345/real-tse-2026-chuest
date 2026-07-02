#!/usr/bin/env python3
"""Per-corpus RE-USE SIM-gate routing (parallelizable). Args: sp corpus.
Per utt: sim_raw=cos(crown,enrol), sim_enh=cos(RE-USE,enrol). keep RE-USE if sim_enh>=sim_raw-0.05 else crown.
Materialize selected wav at the crown's mapped path + copy tse_audio_mapping.csv (trap 4: scorer reads mapped path).
Run in REAL-T env, CUDA_VISIBLE_DEVICES="" (CPU, parallel across corpora)."""
import sys, os, csv, glob, shutil, numpy as np, warnings
warnings.filterwarnings('ignore')
import wespeakerruntime as wsr
import onnxruntime as _ort
_ORIG_IS = _ort.InferenceSession
def _capped_is(*a, **k):
    so = k.pop('sess_options', None) or _ort.SessionOptions()
    so.intra_op_num_threads = 2; so.inter_op_num_threads = 1
    return _ORIG_IS(*a, sess_options=so, **k)
_ort.InferenceSession = _capped_is  # cap onnx threads (avoid 224-core thrash under parallelism)
sp, corpus = sys.argv[1], sys.argv[2]
chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 0
nch = int(sys.argv[4]) if len(sys.argv) > 4 else 1
MARGIN = 0.05
lang = 'chs' if corpus in ('AliMeeting', 'unseen_CN', 'AISHELL-4') else 'en'
m = wsr.Speaker(lang=lang, intra_op_num_threads=2)
def emb(p): return np.asarray(m.extract_embedding(p), dtype=np.float64).squeeze()
def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
ROOT = f'/data/realtse_step1/eval_data/REAL-T-eval{sp[-1]}'
mapping = {r['utterance']: r['path'] for r in csv.DictReader(open(ROOT + '/mapping.csv'))}
enrol_of = {f"{r['mixture_utterance']}-{r['enrolment_speakers_utterance']}": r['enrolment_speakers_utterance']
            for r in csv.DictReader(open(f'{ROOT}/{sp}/{corpus}_meta.csv'))}
crown_dir = f'/data/realtse_step1/ep16_bestof_stage/{sp}/ep16_enh_6way/{corpus}/wav'
reuse_dir = f'/data/realtse_step1/reuse_out/{sp}/{corpus}/wav'
SUBC = f'/data/realtse_step1/submission_reuse_routed/{sp}/ep16_enh_6way/{corpus}'
os.makedirs(SUBC + '/wav', exist_ok=True)
rows = []; ecache = {}
for cw in sorted(glob.glob(crown_dir + '/*.wav'))[chunk::nch]:
    b = os.path.basename(cw); utt = b[:-4]; rw = f'{reuse_dir}/{b}'
    eu = enrol_of.get(utt); ep = mapping.get(eu) if eu else None
    if not (os.path.exists(rw) and ep and os.path.exists(ep)):
        shutil.copy(cw, f'{SUBC}/wav/{b}'); rows.append((sp, corpus, utt, '', '', 'raw_fallback')); continue
    if ep not in ecache: ecache[ep] = emb(ep)
    e = ecache[ep]; sr_ = cos(emb(cw), e); se = cos(emb(rw), e)
    keep = se >= sr_ - MARGIN
    shutil.copy(rw if keep else cw, f'{SUBC}/wav/{b}')
    rows.append((sp, corpus, utt, sr_, se, 'reuse' if keep else 'crown'))
mapcsv = f'/data/realtse_step1/ep16_bestof_stage/{sp}/ep16_enh_6way/{corpus}/tse_audio_mapping.csv'
if chunk == 0 and os.path.exists(mapcsv): shutil.copy(mapcsv, f'{SUBC}/tse_audio_mapping.csv')
with open(f'/data/realtse_step1/reuse_route_{sp}_{corpus}_c{chunk}.csv', 'w', newline='') as fo:
    w = csv.writer(fo); w.writerow(['split', 'corpus', 'utterance', 'sim_raw', 'sim_enh', 'chosen']); w.writerows(rows)
good = [r for r in rows if r[3] != '']
nre = sum(1 for r in good if r[5] == 'reuse')
sraw = np.mean([r[3] for r in good]) if good else 0
srouted = np.mean([r[4] if r[5] == 'reuse' else r[3] for r in good]) if good else 0
print(f'{sp}/{corpus} c{chunk}/{nch}: routed {len(rows)} | reuse {nre}/{len(good)} | SIM crown {sraw:.4f} -> routed {srouted:.4f} (d{srouted-sraw:+.4f})', flush=True)
print(f'REUSE_ROUTE_{sp}_{corpus}_c{chunk}_DONE', flush=True)
