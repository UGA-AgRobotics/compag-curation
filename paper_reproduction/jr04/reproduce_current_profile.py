#!/usr/bin/env python3
"""Hash-bound saved-score/point replay for reviewer-full-20260925-v1.

The original verifier, pins and numerical bodies are unchanged. This separately
named profile binds the sanitized companion and repairs only ten model_path
join values in an output-side copy using the recorded exact identity crosswalk.
No models are loaded, fitted, rescored or inferred.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition: raise ValueError(message)


def member(root, name):
    path = root/name
    require(not path.is_symlink() and path.resolve().is_relative_to(root), 'Unsafe input member')
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', type=Path, required=True, help='Extracted EVIDENCES directory from the exact controlled companion')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); root = a.inputs.resolve(strict=True); out = a.out.resolve()
    require(not out.exists() and not out.is_relative_to(root), 'Choose a new output directory outside the evidence')
    profile = json.loads((HERE/'REVIEWER_FULL_20260925_PROFILE.json').read_text())
    for name, expected in profile['code_sha256'].items():
        require(sha(HERE/name)==expected, 'Preserved evaluator changed: '+name)
    inputs = {}
    for name, record in profile['candidate'].items():
        p = member(root, record['member']); require(sha(p)==record['sha256'], 'Incompatible candidate input: '+name); inputs[name]=p
    pins_file = member(root, profile['point_pins_member'])
    require(sha(pins_file)==profile['point_pins_sha256'], 'Incompatible point-input profile')
    point = member(root, profile['point_inputs_member']); pins=json.loads(pins_file.read_text())
    require(len(pins)==54, 'Expected 54 point inputs')
    for name, expected in pins.items():
        p=member(point, name)
        require(p.stat().st_size==expected['bytes'] and sha(p)==expected['sha256'], 'Incompatible point input: '+name)
    # Only after complete preflight may output be created.
    out.mkdir(parents=True); staged=out/'bound_candidate'; staged.mkdir()
    for name,p in inputs.items(): (staged/name).write_bytes(p.read_bytes())
    table=staged/'table4_reconciled_raw_and_weighted.csv'
    with table.open(newline='') as f: rows=list(csv.DictReader(f))
    require(len(rows)==len(profile['table4_path_crosswalk'])==10,'Expected ten checkpoint identities')
    original=[dict(row) for row in rows]
    for row, change in zip(rows,profile['table4_path_crosswalk']):
        require(int(row['round'])==change['round'] and row['model_path']==change['from'], 'Path identity crosswalk mismatch')
        row['model_path']=change['to']
    for before, after in zip(original,rows):
        require({k:v for k,v in before.items() if k!='model_path'}=={k:v for k,v in after.items() if k!='model_path'},'Scientific table cells changed')
    with table.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    for name, expected in profile['adapted_candidate_sha256'].items():
        require(sha(staged/name)==expected,'Output-side binding mismatch: '+name)
    spec=importlib.util.spec_from_file_location('preserved_jr04_metrics', HERE/'verify_JR04_saved_metrics.py')
    metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    # Explicit named-profile data binding. Original source and historical PINS
    # are untouched; every current/adapted byte was checked above.
    metrics.PINS=dict(profile['adapted_candidate_sha256'])
    saved_argv=sys.argv
    try:
        sys.argv=['verify_JR04_saved_metrics.py','--inputs',str(staged),'--out',str(out/'candidate')]
        metrics.main()
    finally: sys.argv=saved_argv
    env=dict(os.environ, MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    commands=[]
    for label,args in [('s3',['run_s3_v1.py','--inputs',str(point),'--pins',str(pins_file),'--out',str(out/'s3')]),
                       ('generated',['make_jr04_outputs.py','--metrics',str(out/'candidate'),'--s3',str(out/'s3'),'--inputs',str(staged),'--out',str(out/'generated')])]:
        command=[sys.executable,'-B',str(HERE/args[0]),*args[1:]]
        with (out/(label+'.log')).open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=1200)
        commands.append({'step':label,'returncode':result.returncode})
        require(result.returncode==0,label+' failed; see output log')
    candidate=json.loads((out/'candidate/JR04_SAVED_SCORE_CHECK.json').read_text())
    points=json.loads((out/'s3/VERIFICATION_RESULT.json').read_text())
    require(candidate['status']=='SAVED_SCORE_AND_HISTORICAL_TABLE_CHECKS_PASS' and points['scientific_verification']=='PASS','Result verification failed')
    for name,record in profile['candidate'].items():require(sha(inputs[name])==record['sha256'],'Input changed during replay')
    require(points['input_files_unchanged'],'Point inputs changed during replay')
    receipt={'status':'PASS','profile':profile['profile'],'profile_sha256':sha(HERE/'REVIEWER_FULL_20260925_PROFILE.json'),
             'scope':'Saved candidate Table 4 (later Table 5), Figure 7 raw/weighted confusion, S2, retained history/curves and separate point S3; not ablation Table 6',
             'candidate':candidate,'point':points,'commands':commands,'new_training':False,'new_model_scores':False,'new_vision_inference':False,'input_bytes_unchanged':True,
             'adaptation':'Only ten model_path identity joins in the output-side table copy; numerical calculation bodies unchanged',
             'historical_point_note':'Canonical S3 remains 551/740 at 0.80. The companion separately explains the older 552 box-only result; this command does not run that separate historical check.'}
    (out/'CURRENT_PROFILE_ACCEPTANCE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('CURRENT_PROFILE_SAVED_REPLAY_PASS')


if __name__=='__main__':main()
