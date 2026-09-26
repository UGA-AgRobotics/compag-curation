#!/usr/bin/env python3
"""Validated JR06 numerical replay only. Does not modify article or close JR06."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();root=a.root.resolve();out=a.out.resolve()
if out.exists():raise FileExistsError(out)
manifest=json.loads((root/'inputs/SCIENCE_INPUTS_MANIFEST.json').read_text())
for r in manifest['files']:
 q=root/'inputs/core'/r['path']
 if q.stat().st_size!=r['bytes'] or hashlib.sha256(q.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Input identity changed: '+r['path'])
out.mkdir(parents=True)
commands=[]
for script,folder in [('verify_effort_and_gt.py','effort'),('replay_mask.py','mask')]:
 cmd=[sys.executable,str(root/'reproduce'/script),'--inputs',str(root/'inputs/core'),'--out',str(out/folder)]
 with (out/(folder+'_launcher.log')).open('w') as f:ret=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=300)
 commands.append({'command':cmd,'return_code':ret.returncode})
 if ret.returncode:raise RuntimeError('Replay failed; see '+str(out/(folder+'_launcher.log')))
for r in manifest['files']:
 if hashlib.sha256((root/'inputs/core'/r['path']).read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Input changed during replay')
result={'status':'PASS_NUMERICAL_REPLAY_ONLY','JR06':'OPEN_AUTHOR_SCOPE_AND_INTEGRATION_PENDING','commands':commands,'original_inputs_unchanged':True}
(out/'REPRODUCTION_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
