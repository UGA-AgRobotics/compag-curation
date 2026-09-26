#!/usr/bin/env python3
"""JR04 S3 v1: new revision reevaluation using TWO original CVAT exports.
Runs the supplied evaluator unmodified. No fitting, segmentation, or generated GT.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path
import xml.etree.ElementTree as ET
from importlib.metadata import version
import numpy as np
import pandas as pd
CARDS=['IMG_9406','IMG_9459','IMG_9460','IMG_9486','IMG_9487']
EXPECTED_COUNTS=dict(zip(CARDS,[228,130,124,129,129]))
EXPECTED_SETS={
 'IMG_9406':[180,228,0,48,9,10,4,15],
 'IMG_9459':[109,130,0,21,4,4,1,7],
 'IMG_9460':[99,124,0,25,9,8,3,14],
 'IMG_9486':[114,127,2,13,3,0,0,3],
 'IMG_9487':[120,128,1,8,2,3,1,4],
}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,o):Path(p).write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--pins',type=Path,required=True);a=ap.parse_args()
 inp=a.inputs.resolve();out=a.out.resolve();e=inp/'E2E';ev=inp/'eval_points_coverage_v3.py'
 if out.exists() or out==inp or inp in out.parents:raise SystemExit('Use new output directory outside inputs.')
 pins=json.loads(a.pins.read_text())
 if not pins: raise ValueError('Empty input pins')
 for rel,info in pins.items():
  p=inp/rel
  if not p.is_file() or p.stat().st_size!=info['bytes'] or sha(p)!=info['sha256']:
   raise ValueError('Input identity mismatch: '+rel)
 if sha(ev)!='a11b4a3d75ae2ed3e3dee800cbbb193141e59b42b7bad62d4e54448c3ae6c481':
  raise ValueError('Wrong evaluator bytes')
 out.mkdir(parents=True)
 from datetime import datetime, timezone
 write(out/'PROTOCOL_BEFORE_RUN.json',dict(evaluation_id='JR04_S3_REEVALUATION_v1',
   execution_started_utc=datetime.now(timezone.utc).isoformat(),
   author_authorization='User explicitly authorized versioned S3 reevaluation and documented correction in this conversation.',
   cards=CARDS,thresholds=[0.2,0.5,0.8],stage='xgb',score_col='xgb_p',nms_iou=0.5,
   max_area_frac=None,use_p_fused=False,matching='inside any polygon, existing bbox fallback; no one-to-one',
   mean_eligible_count='raw xgb_p >= threshold, per card BEFORE NMS; workload proxy only',
   runner_sha256=sha(__file__),evaluator_sha256=sha(ev),input_pins_sha256=sha(a.pins),
   historical_552_origin='UNKNOWN; this is not a recovered historical execution',
   no_training=True,no_new_model_scoring=True,no_vision_inference=True))
 before={str(p.relative_to(inp)):sha(p) for p in inp.rglob('*') if p.is_file()}
 sources=[inp/'gt/annotations.xml',inp/'gt/E2E_9460_9486.xml']
 seen={};bindings=[]
 for src in sources:
  tree=ET.parse(src).getroot()
  for im in tree.findall('image'):
   name=im.attrib['name'];card=Path(name).stem
   if card not in EXPECTED_COUNTS:raise AssertionError('Unexpected image '+name)
   if card in seen:raise AssertionError('Duplicate image across XML sources '+name)
   seen[card]=src
   pts=[];labels=[];point_sources=[]
   for node in im.findall('points'):
    label=node.attrib.get('label','').strip()
    if label!='cj':raise AssertionError('Unexpected point class '+label)
    for token in node.attrib.get('points','').split(';'):
     if token.strip():
      xy=tuple(map(float,token.split(',')));assert len(xy)==2
      pts.append(xy);labels.append(label);point_sources.append(node.get('source'))
   arr=np.asarray(pts,dtype=np.float32).reshape(-1,2)
   old=pd.read_csv(e/f'eval_{card}_xgb05_m20_reviewability/points_detail__xgb.csv')
   assert len(arr)==EXPECTED_COUNTS[card]==len(old)
   assert np.array_equal(old['gt_idx'].values,np.arange(len(arr)))
   assert old['image'].eq(name).all()
   same=np.array_equal(arr,old[['px','py']].to_numpy(np.float32));assert same,card+' point source mismatch'
   w,h=int(im.get('width')),int(im.get('height'))
   assert np.isfinite(arr).all() and (arr>=0).all() and (arr[:,0]<w).all() and (arr[:,1]<h).all()
   ti=pd.read_csv(e/card/'tile_info/tiles_index.csv')
   assert ti.orig_w.eq(w).all() and ti.orig_h.eq(h).all()
   bindings.append({'card':card,'source_xml':str(src.relative_to(inp)),'source_xml_sha256':sha(src),
    'points':len(arr),'width':w,'height':h,'label':'cj','all_points_source_manual':all(x=='manual' for x in point_sources),
    'coordinates_and_order_equal_after_evaluator_float32':bool(same),
    'duplicate_coordinates':len(arr)-len(set(map(tuple,arr))),
    'saved_csv':str((e/f'eval_{card}_xgb05_m20_reviewability/points_detail__xgb.csv').relative_to(inp)),
    'status':'ORIGINAL_XML_VERIFIED'})
 assert set(seen)==set(CARDS)
 bindings.sort(key=lambda x:x['card']);write(out/'GT_SOURCE_BINDING.json',{'status':'PASS','source_points_verified':sum(x['points'] for x in bindings),'cards':bindings})
 pd.DataFrame(bindings).to_csv(out/'GT_SOURCE_BINDING.csv',index=False)
 thresholds=[.2,.5,.8]
 combined={};comparisons={};sweeps=[];per_card_sweep=[];invocations=[]
 for tau in thresholds:
  perpoints=[];persumm=[]
  for batch,src in enumerate(sources,1):
   cards=[c for c in CARDS if seen[c]==src];d=out/f'xml_batch{batch}_tau{int(tau*100):03d}'
   cmd=[sys.executable,'-B',str(ev),'--cvat_xml',str(src),'--images',','.join(cards),
    '--stage','xgb','--score_col','xgb_p','--thr_xgb',str(tau),'--al_margin','.2','--nms_iou','.5',
    '--det_policy','hybrid','--det_thr','.5','--yolo_conf_thr','.2','--yolo_iou_thr','.6','--out_dir',str(d)]
   for card in cards:
    cmd += ['--pred_csv',str(e/card/'run_xgb_recall/detections.csv'),
            '--tile_index',str(e/card/'tile_info/tiles_index.csv'),
            '--al_candidates_csv',str(e/card/'run_xgb_recall/al_candidates.csv')]
   t=time.time()
   env=os.environ.copy();env['MPLBACKEND']='Agg';env['PYTHONDONTWRITEBYTECODE']='1'
   with (out/(d.name+'.log')).open('w') as log:
    r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=180)
   write(out/(d.name+'_INVOCATION.json'),{'argv':cmd,'return_code':r.returncode,'elapsed_seconds':time.time()-t,'evaluator_sha256':sha(ev),'xml_sha256':sha(src),
       'note':'New evaluation of archived proposals using supplied original XML; use_p_fused=False. No model inference or fitting.'})
   if r.returncode:raise RuntimeError('Evaluator failed: '+d.name)
   invocations.append({'run':d.name,'return_code':r.returncode})
   for name in ['summary__xgb.csv','points_detail__xgb.csv','overall__xgb.json']:
    assert (d/name).is_file() and (d/name).stat().st_size>0
   perpoints.append(pd.read_csv(d/'points_detail__xgb.csv'));persumm.append(pd.read_csv(d/'summary__xgb.csv'))
  pp=pd.concat(perpoints).sort_values(['image','gt_idx']).reset_index(drop=True)
  ss=pd.concat(persumm).sort_values('image').reset_index(drop=True)
  assert len(pp)==740 and len(ss)==5
  assert not pp.duplicated(['image','gt_idx']).any()
  assert set(ss.image)=={c+'.jpg' for c in CARDS}
  for c in CARDS:
   det=pd.read_csv(e/c/'run_xgb_recall/detections.csv')
   ti=pd.read_csv(e/c/'tile_info/tiles_index.csv')
   al=pd.read_csv(e/c/'run_xgb_recall/al_candidates.csv')
   assert not det.duplicated(['image','id']).any()
   assert not ti.tile_name.duplicated().any()
   assert set(det.image)<=set(ti.tile_name)
   assert np.isfinite(det.xgb_p).all() and det.xgb_p.between(0,1).all()
   assert len(al)==50 and not al.duplicated(['image','id']).any()
   assert set(zip(al.image,al.id))<=set(zip(det.image,det.id))
   subset=pp[pp.image.eq(c+'.jpg')];summary=ss[ss.image.eq(c+'.jpg')].iloc[0]
   assert len(subset)==EXPECTED_COUNTS[c]
   assert int(subset.covered_stage.sum())==int(summary.covered_points)
   per_card_sweep.append(dict(threshold=tau,card=c,gt_points=len(subset),
     covered_points=int(summary.covered_points),recall=float(summary.recall_cov),
     pre_nms_candidates=int((det.xgb_p>=tau).sum()),post_nms_instances=int(summary.n_pred_instances),
     proposal_covered_points=int(subset.covered_all.sum())))
  pp.to_csv(out/f'ALL_ORIGINAL_XML_tau{int(tau*100):03d}_points.csv',index=False)
  ss.to_csv(out/f'ALL_ORIGINAL_XML_tau{int(tau*100):03d}_summary.csv',index=False)
  combined[tau]=pp
  sweeps.append({'threshold':tau,'covered_points':int(ss.covered_points.sum()),'gt_points':740,'recall':float(ss.covered_points.sum()/740),'post_nms_instances':int(ss.n_pred_instances.sum()),
                 'pre_nms_candidates_total':sum(x['pre_nms_candidates'] for x in per_card_sweep if x['threshold']==tau),'mean_pre_nms_candidates_per_card':sum(x['pre_nms_candidates'] for x in per_card_sweep if x['threshold']==tau)/5,'status':'NEW_REVISION_REEVALUATION_OF_SAVED_PROPOSALS','evaluation_id':'JR04_S3_REEVALUATION_v1'})
  if tau==.5:
   for c in CARDS:
    old=pd.read_csv(e/f'eval_{c}_xgb05_m20_reviewability/points_detail__xgb.csv')
    new=pp[pp.image.eq(c+'.jpg')].reset_index(drop=True)
    assert list(new.columns)==list(old.columns)
    equal={col:np.array_equal(new[col].values,old[col].values,equal_nan=True) if np.issubdtype(old[col].dtype,np.number) else new[col].equals(old[col]) for col in old.columns}
    comparisons[c]={'rows':len(old),'columns_equal':equal,'all_columns_equal':all(equal.values())}
    assert all(equal.values()),str(comparisons[c])
   rows=[]
   for c in CARDS:
    p=pp[pp.image.eq(c+'.jpg')];d=p[(p.covered_stage==0)&(p.covered_all==1)]
    w=d.any_within_margin.eq(1);s=d.any_in_al_topk.eq(1)
    row={'card':c,'gt_points':len(p),'final_covered':int(p.covered_stage.sum()),'proposal_covered':int(p.covered_all.sum()),
      'proposal_misses':int((p.covered_all==0).sum()),'decision_misses':len(d),'raw_xgb_margin':int(w.sum()),'saved_top50':int(s.sum()),
      'intersection':int((w&s).sum()),'union':int((w|s).sum()),'margin_only':int((w&~s).sum()),'shortlist_only':int((s&~w).sum()),'neither':int((~s&~w).sum())}
    assert [row[k] for k in ['final_covered','proposal_covered','proposal_misses','decision_misses','raw_xgb_margin','saved_top50','intersection','union']]==EXPECTED_SETS[c]
    rows.append(row)
   total={'card':'TOTAL',**{k:sum(r[k] for r in rows) for k in rows[0] if k!='card'}}
   rows.append(total);pd.DataFrame(rows).to_csv(out/'JR01_REGRESSION_tau050.csv',index=False)
 write(out/'PER_POINT_REPRODUCTION.json',comparisons)
 pd.DataFrame(sweeps).sort_values('threshold').to_csv(out/'S3_RESULTS.csv',index=False)
 pd.DataFrame(per_card_sweep).to_csv(out/'S3_PER_CARD.csv',index=False)
 changes=[]
 for row,old in zip(sweeps,[668,622,552]):
  changes.append(dict(threshold=row['threshold'],old_printed_hits=old,new_hits=row['covered_points'],
    delta=row['covered_points']-old,old_display=f'{old/740:.3f}',new_display=f"{row['recall']:.3f}",
    historical_cause='UNKNOWN' if old!=row['covered_points'] else 'NO_COUNT_CHANGE'))
 pd.DataFrame(changes).to_csv(out/'S3_CHANGE_LOG.csv',index=False)
 after={str(p.relative_to(inp)):sha(p) for p in inp.rglob('*') if p.is_file()}
 assert before==after
 write(out/'INPUT_MANIFEST.json',{'files':[{'path':k,'size':(inp/k).stat().st_size,'sha256':v} for k,v in before.items()],'all_input_bytes_unchanged':True})
 write(out/'ENVIRONMENT.json',{'python':platform.python_version(),**{x:version(x) for x in ['numpy','pandas','opencv-python','matplotlib','tabulate']}})
 final={'scientific_verification':'PASS','all_five_original_point_sources_verified':True,'points':740,'original_evaluator_unchanged':True,
        'all_12_columns_match_all_740_saved_point_rows_tau050':all(x['all_columns_equal'] for x in comparisons.values()),'table7':total,
        'historical_original_launch_receipt_authenticated':False,'GT_biological_completeness_reannotated':False,
        'input_files_unchanged':True,'new_training':False,'new_vision_inference':False,
        'threshold_080_disposition':'VERSIONED_REEVALUATION_APPROVED_BY_AUTHOR; historical552origin UNKNOWN','evaluation_id':'JR04_S3_REEVALUATION_v1','commands':invocations,'sweep_results':sweeps,'manuscript_integration':'SEPARATE_CHECK'}
 write(out/'VERIFICATION_RESULT.json',final);print(json.dumps(final,indent=2))
if __name__=='__main__':main()
