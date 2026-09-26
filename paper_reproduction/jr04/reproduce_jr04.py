#!/usr/bin/env python3
"""One entry point for JR04 numerical tables and figures.
Run in a new output directory, with the private bundle's pinned original inputs.
No fitting, model loading, fresh probabilities or image inference is performed.
Manuscript integration is a separate hash-checked patch; this command does not edit it.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys,hashlib
from pathlib import Path

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 inp=a.inputs.resolve();out=a.out.resolve();here=Path(__file__).resolve().parent
 if out.exists() or inp in out.parents:raise ValueError('Output must be new and outside inputs')
 out.mkdir(parents=True)
 steps=[('candidate',[sys.executable,'-B',str(here/'verify_JR04_saved_metrics.py'),'--inputs',str(inp/'candidate'),'--out',str(out/'candidate')]),
        ('s3',[sys.executable,'-B',str(here/'run_s3_v1.py'),'--inputs',str(inp/'point'),'--pins',str(inp/'POINT_INPUT_PINS.json'),'--out',str(out/'s3')]),
        ('generated',[sys.executable,'-B',str(here/'make_jr04_outputs.py'),'--metrics',str(out/'candidate'),'--s3',str(out/'s3'),'--inputs',str(inp/'candidate'),'--out',str(out/'generated')])]
 records=[];env=os.environ.copy();env.update(MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1')
 for name,cmd in steps:
  with (out/(name+'.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=1200)
  records.append({'step':name,'argv':cmd,'return_code':r.returncode})
  (out/'COMMANDS.json').write_text(json.dumps(records,indent=2)+'\n')
  if r.returncode:raise RuntimeError(f'{name} failed, rc={r.returncode}; no success certificate')
 cand=json.loads((out/'candidate/JR04_SAVED_SCORE_CHECK.json').read_text())
 s3=json.loads((out/'s3/VERIFICATION_RESULT.json').read_text())
 if cand['status']!='SAVED_SCORE_AND_HISTORICAL_TABLE_CHECKS_PASS' or s3['scientific_verification']!='PASS':raise RuntimeError('Subcheck failed')
 result={'status':'JR04_NUMERICAL_REPRODUCTION_PASS','candidate_comparisons':cand['metric_comparisons'],
 'historical_table_comparisons':cand['historical_metric_comparisons'],'s3':s3['sweep_results'],
 'historical_552_cause':'UNKNOWN','new_model_inference_or_training':False,
 'scope':'Numerical output reproduction only. Does not independently certify manuscript integration, model selection or final submission readiness.',
 'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [here/n for n in ['reproduce_jr04.py','verify_JR04_saved_metrics.py','run_s3_v1.py','make_jr04_outputs.py']]}}
 (out/'REPRODUCTION_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(result['status'])
if __name__=='__main__':main()
