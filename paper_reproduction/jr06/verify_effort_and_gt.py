#!/usr/bin/env python3
"""Independent raw CSV time/ID and CVAT/COCO correspondence checks; no model imports.
Requires Python standard library only. No historical timing script is executed.
"""
from __future__ import annotations
import argparse,collections,csv,hashlib,json,math
from decimal import Decimal as D
from datetime import datetime,timedelta
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_csv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rows):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['empty']);w.writeheader();w.writerows(rows)
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8')
def seconds(delta):return D(delta.days)*86400+D(delta.seconds)+D(delta.microseconds)/1000000
def cvat_time(rows):
 rr=sorted(enumerate(rows,start=2),key=lambda kv:datetime.fromisoformat(kv[1]['timestamp']))
 traces=[];total=D(0);frames=D(0);ngap=0;prev=None;send=D(0);shapes=[];numframe=0
 for line,r in rr:
  t=datetime.fromisoformat(r['timestamp']);scope=r['scope'].lower();dur=D(r['duration'] or '0')/1000
  gap=seconds(t-prev) if prev is not None else D(0)
  retained=gap if 0<gap<=100 else D(0)
  # Same retained algorithm: first event duration sets prev_end; all actual sessions have no frame changes.
  frame=dur if scope=='change:frame' and traces and dur>0 else D(0)
  total+=retained+frame;frames+=frame;ngap+=int(gap>100);numframe+=int(scope=='change:frame')
  if scope=='send:working_time':
   if dur>0:send+=dur
   else:raise ValueError('Unexpected send:working_time layout: inspect source before using fallback')
  if scope=='create:shapes':
   payload=json.loads(r['payload']);ids=[x['id'] for x in payload['shapes']]
   shapes.append({'line':line,'object_type':r['obj_name'],'count':int(r['count']),'ids':ids})
  traces.append({'source_csv_line':line,'scope':scope,'timestamp':r['timestamp'],'gap_s':str(gap),'counted_gap_s':str(retained),'frame_seconds':str(frame),'cumulative_seconds':str(total)})
  prev=t+timedelta(microseconds=int(dur*1000000)) if scope=='change:frame' else t
 scopes=collections.Counter(x['scope'] for x in rows)
 result={'events':len(rows),'active_seconds':str(total),'send_working_seconds':str(send),'frame_seconds':str(frames),'frame_events':numframe,'gaps_over_100s':ngap,'first':rr[0][1]['timestamp'],'last':rr[-1][1]['timestamp'],'first_scope':rr[0][1]['scope'],'last_scope':rr[-1][1]['scope'],'first_to_last_seconds':str(seconds(datetime.fromisoformat(rr[-1][1]['timestamp'])-datetime.fromisoformat(rr[0][1]['timestamp']))),'identical_duplicate_rows':len(rows)-len({json.dumps(x,sort_keys=True) for x in rows}),'jobs':sorted({x['job_id'] for x in rows}),'accounts':sorted({x['user_name'] for x in rows}),'draw_events':scopes['draw:object'],'delete_events':scopes['delete:object'],'created_rectangle_ids':len({v for x in shapes if x['object_type']=='rectangle' for v in x['ids']}),'created_count_fields':sum(x['count'] for x in shapes),'scope_counts':dict(scopes),'shape_records':shapes}
 return result,traces

EXCLUDED={'auto_accept','bulk_accept_rest','bulk_accept','accept_rest','as_is'}
ALLOWED={'accept','flip','sus_accept','sus_flip','skip'}
def review_time(rows,card,infer):
 if any(card not in r['image'] for r in rows):raise ValueError('Different card in review source')
 rr=sorted(enumerate(rows,start=2),key=lambda kv:datetime.strptime(kv[1]['timestamp'],'%Y%m%d%H%M%S'))
 counts=collections.Counter(r['timestamp'] for _,r in rr);traces=[];prev=None;total=D(0);weight=D(0);retained=[];explicit=0;inferred_new=0;ngap=0
 for line,r in rr:
  raw=r['action'].lower();act='auto_accept' if infer and counts[r['timestamp']]>=10 else raw
  exclude=act in EXCLUDED;explicit+=int(raw in EXCLUDED);inferred_new+=int(exclude and raw not in EXCLUDED)
  if not exclude and act not in ALLOWED:raise ValueError('Unclassified action: '+act)
  t=datetime.strptime(r['timestamp'],'%Y%m%d%H%M%S');gap=None;contrib=D(0)
  if not exclude:
   gap=seconds(t-prev) if prev else D(0);contrib=gap if 0<gap<=100 else D(0);total+=contrib;prev=t;ngap+=int(gap>100)
   weight+=D(r['review_weight']);retained.append((r,act))
  traces.append({'source_csv_line':line,'image':r['image'],'candidate_id':r['id'],'timestamp':r['timestamp'],'raw_action':raw,'effective_action':act,'excluded':int(exclude),'gap_from_previous_manual_s':'' if gap is None else str(gap),'counted_seconds':str(contrib),'cumulative_seconds':str(total)})
 keys=[(r['image'],r['id']) for r in rows];ac=collections.Counter(a for _,a in retained)
 return {'rows':len(rows),'manual_actions':len(retained),'action_counts':dict(ac),'active_seconds':str(total),'weight_sum':str(weight),'excluded_rows':len(rows)-len(retained),'explicit_exclusions':explicit,'additional_inferred_exclusions':inferred_new,'duplicate_keys':len(keys)-len(set(keys)),'identical_duplicate_rows':len(rows)-len({json.dumps(x,sort_keys=True) for x in rows}),'gaps_over_100s':ngap,'first_retained':retained[0][0]['timestamp'] if retained else None,'last_retained':retained[-1][0]['timestamp'] if retained else None,'logged_settings':{k:sorted({r[k] for r in rows}) for k in ['det_policy','thr_xgb','use_xgb','use_yolo','det_thr','thr_yolo','thr_iou']}},traces

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();b=a.inputs.resolve();out=a.out.resolve()
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);checks=[]
 def require(label,ok,details=None):
  checks.append({'check':label,'pass':bool(ok),'details':details})
  if not ok:raise ValueError(label)
 def same_num(label,x,y):require(label,math.isclose(float(x),float(y),rel_tol=0,abs_tol=2e-9),{'recomputed':str(x),'reference':str(y)})
 summary_rows=[];evidence={};identities=[]
 for card in ['9459','9486','9487']:
  e=b/f'effort/{card}';evp=e/('9486_events_bbox_test.csv' if card=='9486' else 'events_bbox_test.csv');rp=e/'review_labels.csv';sp=e/f'IMG_{card}/effort_summary.csv';sr={r['condition']:r for r in load_csv(sp)}
  raw=load_csv(rp);cv,tr=cvat_time(load_csv(evp));rv,rr=review_time(raw,'IMG_'+card,True);alt_mode,_=review_time(raw,'IMG_'+card,False)
  require('primary_burst_rule_irrelevant_'+card,rv['active_seconds']==alt_mode['active_seconds'] and rv['manual_actions']==alt_mode['manual_actions'] and rv['excluded_rows']==rv['explicit_exclusions'])
  for fld,actual in [('active_seconds',cv['active_seconds']),('active_seconds_algo',cv['active_seconds']),('active_seconds_send',cv['send_working_seconds']),('active_min',D(cv['active_seconds'])/60),('n_events',cv['events']),('n_objects',cv['created_rectangle_ids'])]:same_num(f'bbox_{card}_{fld}',actual,sr['bbox'][fld])
  require('box_count_draw_minus_delete_'+card,cv['draw_events']-cv['delete_events']==cv['created_rectangle_ids']==cv['created_count_fields'])
  for fld,actual in [('active_seconds',rv['active_seconds']),('active_min',D(rv['active_seconds'])/60),('n_actions',rv['manual_actions']),('sum_review_weight',rv['weight_sum']),('n_auto_accept',rv['excluded_rows']),('n_rows_total',rv['rows'])]:same_num(f'review_{card}_{fld}',actual,sr['review'][fld])
  for action in sorted(ALLOWED):same_num(f'action_{card}_{action}',rv['action_counts'].get(action,0),sr['review']['n_'+action])
  for kind,result in [('cvat',cv),('review',rv)]:
   require('unique_source_rows_'+kind+card,result['identical_duplicate_rows']==0)
   require('zero_long_gaps_'+kind+card,result['gaps_over_100s']==0)
  require('unique_review_keys_'+card,rv['duplicate_keys']==0)
  old=json.loads((b/f'tasks/T008/audit/IMG_{card}_verification.json').read_text())
  for src,key in [(sp,'summary'),(evp,'cvat'),(rp,'review')]:
   h=sha(src);require('prior_source_identity_'+card+key,old['sources'][key]['sha256']==h);identities.append({'relative_path':str(src.relative_to(b)),'sha256':h,'bytes':src.stat().st_size})
  ratio=D(cv['active_seconds'])/D(rv['active_seconds']);reduction=(1-D(rv['active_seconds'])/D(cv['active_seconds']))*100
  summary_rows.append({'card':'IMG_'+card,'bbox_job_seconds':cv['active_seconds'],'bbox_job_minutes':str(D(cv['active_seconds'])/60),'review_manual_seconds':rv['active_seconds'],'review_manual_minutes':str(D(rv['active_seconds'])/60),'event_supported_box_count':cv['created_rectangle_ids'],'manual_review_actions':rv['manual_actions'],'excluded_auto_rows':rv['excluded_rows'],'review_candidate_rows':rv['rows'],'time_ratio':str(ratio),'descriptive_percent_difference':str(reduction),'thr_xgb_in_review_log':','.join(rv['logged_settings']['thr_xgb']),'final_quality_established':'NO'})
  evidence['IMG_'+card]={'cvat':cv,'review':rv,'explicit_only':alt_mode,'saved_summary_identity':sha(sp)}
  write_csv(out/f'IMG_{card}_cvat_receipt.csv',tr);write_csv(out/f'IMG_{card}_review_receipt.csv',rr)
 altp=b/'effort/9486/review_labels_9486.csv';alternate_rows=load_csv(altp);alternate,tr=review_time(alternate_rows,'IMG_9486',True);ex,_=review_time(alternate_rows,'IMG_9486',False)
 selected={(r['image'],r['id']):r for r in load_csv(b/'effort/9486/review_labels.csv')};oldkeys={(r['image'],r['id']):r for r in alternate_rows}
 require('alternate_candidate_keyset',set(selected)==set(oldkeys));differences=[]
 for key in sorted(selected):
  ar=oldkeys[key];sr=selected[key]
  if ar['human_label']!=sr['human_label']:differences.append({'image':key[0],'candidate_id':key[1],'selected_human_label':sr['human_label'],'alternate_human_label':ar['human_label']})
 write_csv(out/'IMG_9486_alternate_label_differences.csv',differences);write_csv(out/'IMG_9486_alternate_review_receipt.csv',tr)
 oldrec=json.loads((b/'tasks/T008/audit/IMG_9486_alternate_verification.json').read_text())
 require('alternate_source_identity',sha(altp)==oldrec['source']['sha256'])
 same_num('alternate_active_seconds',alternate['active_seconds'],oldrec['independent_review']['active_seconds']);same_num('alternate_actions',alternate['manual_actions'],oldrec['independent_review']['retained_manual_actions'])
 evidence['IMG_9486_alternate']={'review':alternate,'without_burst_inference':ex,'label_differences':len(differences),'timestamps_all_differ':all(selected[k]['timestamp']!=oldkeys[k]['timestamp'] for k in selected),'selected_by':'Existing manuscript summary points to review_labels.csv; no new best-session selection','original_prediction_or_quality_approval':'NOT_ESTABLISHED'}
 # GT must come from the original source bundle, not results derived from predictions.
 gtp=b/'mask_study/input/final_gt/3carts_poly_annotations_QAed_v2.zip'
 with zipfile.ZipFile(gtp) as z:
  require('GT_inner_members',set(z.namelist())=={'instances_default.json','annotations.xml'})
  gt=json.loads(z.read('instances_default.json'));xml=ET.fromstring(z.read('annotations.xml'))
  frozen=b/'mask_study/results/run_20260803_184228_8PwhfM/frozen_inputs'
  for n in z.namelist():require('GT_frozen_identity_'+n,z.read(n)==(frozen/n).read_bytes())
 images={i['id']:i for i in gt['images']};cats={c['id']:c['name'].casefold() for c in gt['categories']};records=[];total=0
 for xi in xml.findall('image'):
  name=xi.attrib['name'];co=[i for i in images.values() if i['file_name']==name];require('GT_unique_image_'+name,len(co)==1);im=co[0]
  require('GT_dimensions_'+name,int(xi.attrib['width'])==im['width'] and int(xi.attrib['height'])==im['height'])
  ann=[a for a in gt['annotations'] if a['image_id']==im['id']];xp=xi.findall('polygon');require('GT_counts_'+name,len(ann)==len(xp))
  for aa,xx in zip(ann,xp):
   coords=[D(v) for pair in xx.attrib['points'].split(';') for v in pair.split(',')]
   require('GT_label_'+str(aa['id']),xx.attrib['label'].casefold()==cats[aa['category_id']]=='cj')
   require('GT_ordered_vertices_'+str(aa['id']),len(aa['segmentation'])==1 and coords==[D(str(v)) for v in aa['segmentation'][0]])
   records.append({'image':name,'annotation_id':aa['id'],'vertices':len(coords)//2,'xml_coco_vertices_equal':True,'source':xx.attrib.get('source','')});total+=1
 require('GT_unique_annotation_ids',len({a['id'] for a in gt['annotations']})==len(gt['annotations'])==total)
 qa=load_csv(b/'mask_study/qa/GT_QA_CHECKLIST_completed.csv');require('QA_status_and_notes',all(r['status']=='DONE' and r['notes'].strip() for r in qa))
 write_csv(out/'GT_XML_COCO_CORRESPONDENCE.csv',records);write_csv(out/'EFFORT_REPRODUCED.csv',summary_rows)
 dump(out/'EFFORT_RECONSTRUCTION.json',evidence);dump(out/'SOURCE_IDENTITIES.json',identities)
 dump(out/'GT_AND_QA_CHECK.json',{'total_GT':total,'card_counts':dict(collections.Counter(r['image'] for r in records)),'ordered_polygon_vertices_match':True,'qa_rows':len(qa),'qa_status_counts':dict(collections.Counter(r['status'] for r in qa)),'human_reinspection_of_images':'NOT_RUN','final_timed_session_exports':'NOT_SUPPLIED','ground_truth_is_not_final_review_output':True})
 dump(out/'CHECKS.json',{'status':'PASS_SCOPED_NUMERICAL_AND_SOURCE_CHECKS','JR06_status':'OPEN_AUTHOR_SCOPE_PENDING','check_count':len(checks),'checks':checks})
 print(json.dumps({'status':'PASS_SCOPED_NUMERICAL_AND_SOURCE_CHECKS','checks':len(checks),'gt_polygons':total,'effort_seconds':[[r['card'],r['bbox_job_seconds'],r['review_manual_seconds']] for r in summary_rows],'alternate_label_differences':len(differences)},indent=2))
if __name__=='__main__':main()
