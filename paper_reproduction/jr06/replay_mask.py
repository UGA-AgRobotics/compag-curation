#!/usr/bin/env python3
"""Run the retained evaluator without editing it, on hash-bound retained inputs.
Creates a fresh scratch project. No model, inference, overlays, or source mutation.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, pathlib, shutil, subprocess, sys, time
P=pathlib.Path
SOURCE_SHA='9a7cc3afaf0128a70b8e8ec02d5bb8fa187692fa637bd5121447ff37ad776af4'
RECORDED_RUN='run_20260803_184228_8PwhfM'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--inputs',required=True,type=P);ap.add_argument('--out',required=True,type=P);a=ap.parse_args()
 b=a.inputs.resolve();out=a.out.resolve()
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True)
 rec=b/'mask_study/results'/RECORDED_RUN
 source=b/'mask_study/code/run_revision_instance_mask_evaluation.sh'
 if sha(source)!=SOURCE_SHA:raise ValueError('Unexpected evaluator identity')
 recorded=list(csv.DictReader((rec/'input_manifest.tsv').open(),delimiter='\t'))
 project=out/'scratch_project';project.mkdir()
 bindings=[]
 for row in recorded:
  rel=row['path']
  # Original images affect overlays only and are intentionally not included.
  if '/full_images/' in rel:
   bindings.append(dict(row,disposition='NOT_USED_OVERLAY_ONLY'));continue
  if '/frozen_inputs/' in rel:continue  # Evaluator extracts the checked GT ZIP below.
  if '/detections.csv' in rel or '/tiles_index.csv' in rel:
   card=rel.split('/E2E/')[1].split('/')[0].replace('IMG_',''); src=b/'effort'/card/P(rel).name
  elif rel.startswith('revision/01_instance_mask_evaluation/'):
   src=b/'mask_study'/rel.split('revision/01_instance_mask_evaluation/',1)[1]
  else:raise ValueError(f'Unexpected input: {rel}')
  h=sha(src)
  if h!=row['sha256'] or src.stat().st_size!=int(row['size_bytes']):raise ValueError(f'Wrong source identity: {src}')
  dest=project/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
  bindings.append(dict(row,disposition='MATCHED_COPIED',source_relative=str(src.relative_to(b))))
 work=project/'revision/01_instance_mask_evaluation';script=work/'code'/source.name
 env=os.environ.copy();env.update(PROJECT_ROOT=str(project),REVISION_ROOT=str(project/'revision'),WORK_ROOT=str(work),CODE_DIR=str(work/'code'),INPUT_DIR=str(work/'input/final_gt'),QA_DIR=str(work/'qa'),RESULTS_ROOT=str(out/'evaluator_runs'),PYTHON_BIN=sys.executable,PREDICTION_ROOT=str(project/'Shared/maskout_tile/test_set/E2E'),IMAGE_ROOT='',GT_INPUT=str(work/'input/final_gt/3carts_poly_annotations_QAed_v2.zip'),QA_CHECKLIST=str(work/'qa/GT_QA_CHECKLIST_completed.csv'),ALLOW_DRAFT_GT='0')
 protocol={'kind':'JR06_MASK_REPLAY_v1','evaluator_sha256':SOURCE_SHA,'inputs':bindings,'scope':'Original shell/evaluator, unchanged. Fresh local run, no model or vision inference. Image overlays intentionally not requested.','command':['bash',str(script)],'environment':{k:env[k] for k in ['PROJECT_ROOT','WORK_ROOT','RESULTS_ROOT','PYTHON_BIN','PREDICTION_ROOT','GT_INPUT','QA_CHECKLIST','IMAGE_ROOT','ALLOW_DRAFT_GT']}}
 dump(out/'PROTOCOL_BEFORE_RUN.json',protocol)
 start=time.monotonic()
 with (out/'stdout.log').open('wb') as so,(out/'stderr.log').open('wb') as se:
  proc=subprocess.run(['bash',str(script)],env=env,cwd=project,stdout=so,stderr=se,timeout=240)
 dump(out/'INVOCATION_RESULT.json',{'return_code':proc.returncode,'elapsed_seconds':time.monotonic()-start,'evaluator_sha256_after':sha(script)})
 if proc.returncode:raise RuntimeError(f'Evaluator failed {proc.returncode}; see logs')
 dirs=list((out/'evaluator_runs').glob('run_*'))
 if len(dirs)!=1:raise ValueError('Expected exactly one fresh run')
 gen=dirs[0]
 for n in ['instances_default.json','annotations.xml']:
  expected=next(r for r in recorded if '/frozen_inputs/' in r['path'] and r['path'].endswith('/'+n))
  if sha(gen/'frozen_inputs'/n)!=expected['sha256']:raise ValueError(f'GT inner identity mismatch: {n}')
 checks=[]
 import pandas as pd
 import numpy as np
 for reference in sorted(rec.glob('*.csv')):
  got=gen/reference.name
  if not got.is_file():raise FileNotFoundError(got)
  x=pd.read_csv(reference);y=pd.read_csv(got)
  if list(x.columns)!=list(y.columns) or x.shape!=y.shape:raise ValueError('Shape/schema mismatch: '+got.name)
  maxerr=0.;num=0
  for c in x:
   if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c]):
    aa=x[c].to_numpy(float);bb=y[c].to_numpy(float)
    if not np.array_equal(np.isnan(aa),np.isnan(bb)):raise ValueError('NaN mismatch '+c)
    if not np.allclose(aa,bb,atol=1e-12,rtol=0,equal_nan=True):raise ValueError(f'Numeric mismatch {got.name}/{c}')
    e=np.abs(aa-bb);finite=e[np.isfinite(e)];maxerr=max(maxerr,float(finite.max(initial=0)));num+=int(np.isfinite(aa).sum())
   elif not x[c].fillna('<NA>').astype(str).equals(y[c].fillna('<NA>').astype(str)):raise ValueError(f'Identity mismatch {got.name}/{c}')
  checks.append({'file':got.name,'rows':len(x),'columns':len(x.columns),'numeric_values_compared':num,'maximum_absolute_difference':maxerr,'byte_equal':sha(reference)==sha(got),'status':'PASS'})
 ref_config=json.loads((rec/'evaluation_config.json').read_text());got_config=json.loads((gen/'evaluation_config.json').read_text())
 if got_config!=ref_config:raise ValueError('Config mismatch')
 original=json.loads((rec/'evaluation_summary.json').read_text());actual=json.loads((gen/'evaluation_summary.json').read_text())
 if actual['primary_result']!=original['primary_result']:raise ValueError('Primary result differs')
 for bind in bindings:
  if bind['disposition']=='MATCHED_COPIED' and sha(b/bind['source_relative'])!=bind['sha256']:raise ValueError('Original input changed')
 out_science=out/'scientific_outputs';out_science.mkdir()
 for n in ['evaluation_summary.json','evaluation_config.json','qa_validation.json','METHODS_AND_DEFINITIONS.txt','input_manifest.tsv']+[x['file'] for x in checks]:shutil.copyfile(gen/n,out_science/n)
 result={'status':'PASS_SAVED_OUTPUT_REPLAY','JR06_status':'OPEN_AUTHOR_SCOPE_PENDING','matching_csv_files':len(checks),'numeric_values_compared':sum(x['numeric_values_compared'] for x in checks),'csv_checks':checks,'configuration_identical':True,'primary_result_identical':True,'original_inputs_unchanged':True,'generated_directory':str(gen),'no_new_ground_truth_annotation':True,'visual_annotation_quality_review':'NOT_RUN','post_review_quality':'NOT_ESTABLISHED','primary_result':actual['primary_result'],'environment':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__}}
 dump(out/'REPLAY_CHECK.json',result)
 print(json.dumps({k:result[k] for k in ['status','matching_csv_files','numeric_values_compared','primary_result_identical']},indent=2))
if __name__=='__main__':main()
