"""Replay published capture-exchange counts from a completed BankSim cache.

No model libraries, fitting, prediction or bootstrap entry points are imported.
Requires Python >=3.11 and NumPy. Output must be a new, separate directory.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def contained(root, rel):
    path = (root/rel).resolve()
    require(path.is_relative_to(root), 'Input receipt escapes its declared directory')
    return path

def capacity(n, q):
    f = Fraction(str(q))
    return (n*f.numerator+f.denominator-1)//f.denominator

def priority(ids):
    return np.array([hashlib.sha256(f'banksim-capacity-v1|{int(i)}'.encode('utf-8')).hexdigest() for i in ids])

def partition(a, b, positives):
    r = len(a & b & positives)
    g = len((b-a) & positives)
    lost = len((a-b) & positives)
    u = len(positives-(a | b))
    return dict(R=r, G=g, L=lost, U=u, TP_before=r+lost, TP_after=r+g, delta_tp=g-lost)

def execute(run, esm3, esm4, report):
    # Verify delivered source-free tables before using them as expected values.
    for root, manifest, kind in [(esm3, 'PACKAGE_MANIFEST.json', 'esm3'), (esm4, 'PACKAGE_MANIFEST.json', 'esm4')]:
        content = load(root/manifest)
        items = content['files'] if isinstance(content, dict) else content
        for item in items:
            name = item.get('package_path', item.get('path'))
            prefix = 'banksim/' if kind == 'esm3' else 'posthoc_revision/'
            if name.startswith(prefix):
                name = name[len(prefix):]
            path = contained(root, name)
            require(sha(path) == item['sha256'], f'{kind} package digest mismatch: {name}')
    protocol = load(run/'protocol/effective.json')
    reference = load(esm3/'reference_protocol/effective.json')
    fields = ['datasets', 'information_sets', 'features', 'preprocessing', 'tuning', 'model_parameters', 'seeds', 'capacity', 'contrasts', 'statistics']
    for key in fields:
        require(protocol[key] == reference[key], f'Scientific protocol mismatch: {key}')
    require(load(run/'protocol/SELECTION.json')['chosen'] == load(esm3/'reference_protocol/SELECTION.json')['chosen'], 'Selected configuration differs; do not force historical agreement')
    receipt = load(run/'protocol/EVALUATION_COMPLETE.json')
    require(receipt['status'] == 'evaluated', 'Evaluation is incomplete')
    require(receipt['freeze_sha256'] == sha(run/'protocol/FREEZE.json'), 'Run freeze link mismatch')
    require(receipt['test_release_sha256'] == sha(run/'protocol/TEST_RELEASE.json'), 'Run test-release link mismatch')
    lineage = []
    for item in receipt['artifacts']:
        path = contained(run, item['path'])
        require(sha(path) == item['sha256'], f'Run artifact mismatch: {item["path"]}')
        lineage.append(dict(item))
    old_prep = load(esm3/'audit/pretraining_PREPARED.json')
    own_prep = load(run/'audit/PREPARED.json')
    for rel in ['features/test_identity.npy', 'preprocessing/B0.json', 'preprocessing/B0H.json', 'preprocessing/B1.json', 'preprocessing/B1H.json']:
        old = next(x for x in old_prep['artifacts'] if x['path'] == rel)
        own = next(x for x in own_prep['artifacts'] if x['path'] == rel)
        require(sha(run/rel) == old['sha256'] == own['sha256'], f'Identity/preprocessing mismatch: {rel}')
    ids = np.load(run/'features/test_identity.npy', allow_pickle=False)
    labels = np.load(run/'predictions/test_labels.npy', allow_pickle=False)
    old_receipt = load(esm3/'reference_protocol/EVALUATION_COMPLETE.json')
    label_hash = next(x['sha256'] for x in old_receipt['artifacts'] if x['path'] == 'predictions/test_labels.npy')
    require(sha(run/'predictions/test_labels.npy') == label_hash, 'Historical label identity differs')
    require(ids.shape == (198311, 2) and len(np.unique(ids[:,0])) == len(ids), 'Invalid source identity')
    require(np.array_equal(np.unique(ids[:,1]), np.arange(126,180)), 'Test boundaries differ')
    require(labels.shape == (len(ids),) and set(np.unique(labels)) == {0,1}, 'Invalid labels')
    require(int(labels.sum()) == 2160, 'Fraud count differs')
    families = ('LightGBM', 'XGBoost')
    seeds = (42,52,62,72,82)
    expected_keys = {f'{f}|{c}|{s}' for f in families for c in ('B0','B0H','B1','B1H') for s in seeds}
    require(set(receipt['score_files']) == expected_keys, 'Not the complete 40 conditions')
    selected, tp_checks, time_checks = {}, 0, 0
    absolute = {(r['family'],r['cell'],int(r['seed'])):r for r in rows(esm3/'results/absolute_all_conditions.csv') if r['segment'] == 'whole'}
    time_rows = {(r['family'],r['cell'],int(r['seed']),int(r['step'])):int(r['tp']) for r in rows(esm4/'derived/time_structure/banksim_step_capture_all_seeds.csv')}
    ties = priority(ids[:,0])
    positives = set(ids[labels == 1,0].tolist())
    for key in sorted(expected_keys):
        family, cell, seed_text = key.split('|')
        seed = int(seed_text)
        rel = receipt['score_files'][key]
        require(any(x['path'] == rel for x in lineage), 'Score not bound to evaluation receipt')
        score = np.load(contained(run, rel), allow_pickle=False)
        require(score.shape == labels.shape and np.isfinite(score).all(), 'Invalid cached margin')
        order = np.lexsort((ids[:,0],ties,-score))
        for q in (0.01,0.03,0.05):
            idx = order[:capacity(len(ids),q)]
            require(int(labels[idx].sum()) == int(absolute[family,cell,seed][f'tp@{q}']), 'Published absolute capture differs')
            tp_checks += 1
            selected[family,cell,seed,q] = set(ids[idx,0].tolist())
            if q == 0.03:
                counts = np.bincount(ids[idx,1]-126, weights=labels[idx], minlength=54).astype(int)
                for step, count in zip(range(126,180), counts, strict=True):
                    require(int(count) == time_rows[family,cell,seed,step], 'Published step-attribution count differs')
                    time_checks += 1
    observed = []
    expected = {(r['family'],r['contrast'],int(r['seed']),float(r['q'])):r for r in rows(esm4/'derived/posthoc_capture_exchange/all_60_pairs.csv')}
    for family in families:
        for before, after, contrast in [('B0','B0H','H0'),('B1','B1H','H1')]:
            for seed in seeds:
                for q in (0.01,0.03,0.05):
                    values = partition(selected[family,before,seed,q], selected[family,after,seed,q], positives)
                    expected_row = expected[family,contrast,seed,q]
                    for name, value in values.items():
                        require(value == int(expected_row[name]), f'Published {name} differs: {family}/{contrast}/{seed}/{q}')
                    require(sum(values[k] for k in ('R','G','L','U')) == len(positives), 'Partition not exhaustive')
                    observed.append(dict(family=family,contrast=contrast,seed=seed,q=q,**values))
    for row in rows(esm4/'derived/posthoc_capture_exchange/means_12.csv'):
        subset = [x for x in observed if x['family']==row['family'] and x['contrast']==row['contrast'] and x['q']==float(row['q'])]
        require(len(subset)==5, 'Incomplete mean')
        for name in ('R','G','L','U','TP_before','TP_after','delta_tp'):
            require(abs(sum(x[name] for x in subset)/5-float(row[name])) <= 1e-9, 'Mean differs beyond fixed absolute tolerance')
    report.update(status='PASSED_LOCAL_CACHE_INTEROPERABILITY', scientific_protocol_fields=fields, cached_margin_vectors=40, absolute_TP_checks=tp_checks, exchange_pairs=len(observed), mean_rows=12, global_selection_step_checks=time_checks, integer_tolerance=0, mean_absolute_tolerance=1e-9, lineage=lineage, new_fits=0, new_model_scores=0, new_bootstrap_draws=0, independent_raw_to_model_rebuild=False)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('run','esm3','esm4','out'):
        parser.add_argument('--'+flag, type=Path, required=True)
    args = parser.parse_args()
    run, esm3, esm4, out = [getattr(args,k).resolve() for k in ('run','esm3','esm4','out')]
    require(not out.exists(), 'Output already exists; do not overwrite a replay receipt')
    require(all(not out.is_relative_to(x) for x in (run,esm3,esm4)), 'Output must be separate from inputs')
    out.mkdir(parents=True)
    report = dict(status='STARTED', run=str(run), esm3=str(esm3), esm4=str(esm4), python=sys.version, numpy=np.__version__, entry_sha256=sha(__file__))
    start = time.perf_counter()
    try:
        execute(run,esm3,esm4,report)
    except Exception as exc:
        report.update(status='FAILED_STOPPED_NO_INPUTS_MODIFIED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['seconds'] = time.perf_counter()-start
        (out/'REPLAY_RECEIPT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(report['status'])

if __name__ == '__main__':
    main()
