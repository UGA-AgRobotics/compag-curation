#!/usr/bin/env python3
"""JR02 bounded verification. No fit/retraining or vision inference.
Read trusted r92 through an allowlisted loader; model/scorer pins target JR09 metadata-only derivatives. Originals stay read-only.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse, ast, csv, gc, hashlib, json, platform, re, sys, warnings
if not __debug__:raise RuntimeError('Run without Python -O; acceptance assertions are required.')
from pathlib import Path
from typing import Any, Optional
from collections import Counter
import numpy as np, pandas as pd, xgboost as xgb, sklearn
from joblib.numpy_pickle import NumpyUnpickler
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, precision_recall_fscore_support
EXPECTED={'model':'88a1be9e5adccbd46ad07de2c702583a70d99d7a59d7572514a12462b1b91554','audit':'c2be6932a56f13117b18e8ed5750dac5f5317fce77ff1ece41ddf948a0ffcee6','snapshot':'1836d704d15f27728e3ba09d7cddf0aea9458049bc66700dc98f14e7111ec29d','scorer':'d9bb7eca020ee3f84bdefd262e827388844a2a591cde9f27e125ab148a913676'}
ALLOWED={('imblearn.pipeline','Pipeline'),('sklearn.impute._base','SimpleImputer'),('imblearn.over_sampling._smote.base','SMOTE'),('sklearn.neighbors._unsupervised','NearestNeighbors'),('sklearn.neighbors._kd_tree','KDTree'),('sklearn.metrics._dist_metrics','EuclideanDistance64'),('xgboost.sklearn','XGBClassifier'),('xgboost.core','Booster'),('numpy','ndarray'),('numpy','dtype'),('numpy.core.multiarray','_reconstruct'),('numpy._core.multiarray','_reconstruct'),('numpy.core.multiarray','scalar'),('numpy._core.multiarray','scalar'),('joblib.numpy_pickle','NumpyArrayWrapper'),('builtins','bytearray'),('builtins','set'),('collections','OrderedDict')}
class Restricted(NumpyUnpickler):
 def find_class(self,mod,name):
  if (mod,name) not in ALLOWED: raise RuntimeError(f'Disallowed {mod}.{name}')
  return super().find_class(mod,name)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ser(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,Path):return str(x)
 raise TypeError(str(type(x)))
def dump(p,x):p.write_text(json.dumps(x,default=ser,indent=2,ensure_ascii=False))
def metrics(y,p,w=None):
 pr,rc,f1,_=precision_recall_fscore_support(y,p>=.5,average='binary',sample_weight=w,zero_division=0)
 return dict(confusion_TN_FP_FN_TP=confusion_matrix(y,p>=.5,sample_weight=w).ravel().tolist(),precision=pr,recall=rc,f1=f1,AP=average_precision_score(y,p,sample_weight=w),ROC_AUC=roc_auc_score(y,p,sample_weight=w))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--snapshot',type=Path,required=True);ap.add_argument('--prior-snapshot',type=Path);ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 O=args.out;O.mkdir(parents=True,exist_ok=False)
 mod=args.model_dir/'cj_ultra_tilesafe_xgb_r92_hybrid.pkl';scorer=args.run_dir/'run_revision_candidate_metrics_model_audit_v3.sh'
 paths=dict(model=mod,audit=args.audit,snapshot=args.snapshot,scorer=scorer)
 ids={k:dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for k,p in paths.items()}
 assert all(ids[k]['sha256']==v for k,v in EXPECTED.items()),ids
 dump(O/'INPUT_IDENTITIES.json',ids)
 print('INPUT IDENTITIES MATCH',flush=True)
 with warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter('always')
  with open(mod,'rb') as f:obj=Restricted(str(mod),f,ensure_native_byte_order=True).load()
 (O/'model_loading_warnings.txt').write_text('\n'.join(str(w.message) for w in ws))
 pipe=obj['pipeline'];imp=pipe.named_steps['imp'];clf=pipe.named_steps['clf'];bst=clf.get_booster();cols=list(obj['features']);gates=[c for c in cols if c.startswith('g_')]
 assert cols==imp.feature_names_in_.tolist() and bst.num_features()==len(cols)==93
 meta=dict(source=ids['model'],features=cols,statistics=imp.statistics_,stored_threshold=obj['threshold'],trees=bst.num_boosted_rounds(),payload_meta=obj.get('meta'),environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,xgboost=xgb.__version__,sklearn=sklearn.__version__),model_modified_or_refitted=False)
 dump(O/'MODEL_SCHEMA_IMPUTER.json',meta)
 # Use exact source cleaning/name functions only; never execute the original shell or its main program.
 shell=scorer.read_text();pytxt=shell.split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]
 tree=ast.parse(pytxt)
 names={'normalize_image_name','original_card','load_and_clean_testset'}
 keep=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 assert len(keep)==3
 env=dict(Path=Path,Any=Any,np=np,pd=pd,re=re,note=print,POSITIVE_CLASS='CJ (label=1)',WEIGHT_MAP={'accept':1.,'flip':1.,'sus_accept':.4,'sus_flip':.4,'skip':0.,'auto_accept':0.},TILE_PATTERNS=[re.compile(r'^(?P<root>.+?)_y\d{1,8}x\d{1,8}$',re.I),re.compile(r'^(?P<root>.+?)_x\d{1,8}_y\d{1,8}$',re.I),re.compile(r'^(?P<root>.+?)(?:__tile-\d+|-tile-\d+)$',re.I)])
 exec(compile(ast.Module(body=keep,type_ignores=[]),'original_scorer_read_only_defs','exec'),env)
 df,clean=env['load_and_clean_testset'](args.audit);dump(O/'AUDIT_POPULATION.json',clean)
 X=df.reindex(columns=cols).copy()
 for c in X:
  if not pd.api.types.is_numeric_dtype(X[c]):X[c]=pd.to_numeric(X[c],errors='coerce')
 X=X.replace([np.inf,-np.inf],np.nan)
 imputation_fallbacks=[]
 def impute(v):
  try:return imp.transform(v)
  except AttributeError as e:
   # Explicit equivalent transform from existing fitted statistics, no fitting/state edits.
   assert not imp.add_indicator and np.isfinite(imp.statistics_).all()
   arr=v.to_numpy(dtype=float,copy=True);m=np.isnan(arr)
   arr[m]=np.take(imp.statistics_,np.where(m)[1])
   imputation_fallbacks.append(repr(e));return arr
 Xt=impute(X)
 assert Xt.shape==X.shape
 # Attempt original predict_proba call; if new wrapper version is incompatible, use stored booster directly.
 with warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter('always')
  try:
   p=pipe.predict_proba(X)[:,1];scoring_path='stored_pipeline.predict_proba';wrapper_error=None
  except Exception as e:
   wrapper_error=repr(e);scoring_path='stored fitted median statistics -> stored booster.predict (wrapper-version compatibility only; no zero-fill)'
   bst.set_param({'device':'cpu','nthread':2});p=bst.predict(xgb.DMatrix(Xt))
 (O/'scoring_warnings.txt').write_text('\n'.join(str(w.message) for w in ws))
 saved=pd.read_csv(args.run_dir/'scored_candidates_r92.csv');keys=['img_folder','image','id'];assert len(df)==len(saved)==10097
 assert df[keys].equals(saved[keys]);assert np.array_equal(df.human_label,saved.human_label);assert np.array_equal(df.review_weight,saved.review_weight)
 target=saved.canonical_score.to_numpy();delta=np.abs(p.astype(float)-target)
 result=dict(rows=len(df),same_identity_order=True,predictor_columns=93,missing_column_names=sorted(set(cols)-set(df.columns)),scoring_path=scoring_path,wrapper_error=wrapper_error,max_abs_decimal_difference=delta.max(),float32_equal_count=int((p==target.astype(np.float32)).sum()),changed_decisions_tau05=int(np.count_nonzero((p>=.5)!=(target>=.5))),raw=metrics(df.human_label,p),weighted=metrics(df.human_label,p,df.review_weight),all_nan_columns=X.columns[X.isna().all()].tolist(),any_nan_columns=X.columns[X.isna().any()].tolist(),complete_numeric_columns=int((~X.isna().any()).sum()),missing_values=int(X.isna().sum().sum()))
 dump(O/'AUDIT_SCORING_SUMMARY.json',result)
 assert result['float32_equal_count']==10097 and result['changed_decisions_tau05']==0
 scored=df[keys+['human_label','action','review_weight']].copy();scored['saved_score']=target;scored['recomputed_score']=p.astype(float);scored['abs_delta']=delta;scored.to_csv(O/'r92_audit_prediction_reconciliation.csv',index=False)
 np.save(O/'r92_audit_scores_float32.npy',p,allow_pickle=False)
 profile=[]
 for i,c in enumerate(cols):
  s=X[c];r=dict(position=i,feature=c,missing=int(s.isna().sum()),finite_unique=int(s.nunique()),nonbinary=int((s.notna()&~s.isin([0,1])).sum()),min=s.min(),max=s.max(),imputer_median=imp.statistics_[i],post_impute_unique=int(np.unique(Xt[:,i]).size))
  profile.append(r)
 pd.DataFrame(profile).to_csv(O/'audit_predictor_profile.csv',index=False)
 print('AUDIT',json.dumps(result,default=ser),flush=True)
 # Distinguish cached operational probability from retrospective r92 score.
 cached=df.xgb_proba.to_numpy();cache_cmp={'rows':len(df),'same_float32':int(np.count_nonzero(cached.astype(np.float32)==p)), 'changed_decisions_at05':int(np.count_nonzero((cached>=.5)!=(p>=.5))),'max_abs_delta':float(np.max(np.abs(cached-p))), 'note':'Different saved scoring paths; not proof of model identity of cached score.'}
 dump(O/'CACHED_VS_RESCORED.json',cache_cmp)
 # Bounded impact check: all-zero gate vector versus verified median-filled audit. This is an explicit diagnostic, NOT a replacement result.
 Xz=np.array(Xt,copy=True);Xz[:,[cols.index(g) for g in gates]]=0
 bst.set_param({'device':'cpu','nthread':2});pz=bst.predict(xgb.DMatrix(Xz))
 diag={'scope':'COUNTERFACTUAL_INPUT_DIAGNOSTIC_NOT_CORRECTED_PERFORMANCE','change':'Only eight audit gate values set to zero instead of verified stored median imputation. No tuning/training/labels used to choose settings. Not a recovered production pathway.','changed_decisions_at05':int(np.count_nonzero((pz>=.5)!=(p>=.5))),'max_probability_delta':float(np.max(np.abs(pz-p))),'raw':metrics(df.human_label,pz),'cached_score_float32_match_count':int(np.count_nonzero(pz==cached.astype(np.float32)))}
 dump(O/'GATE_ZERO_SENSITIVITY_DIAGNOSTIC.json',diag)
 print('GATE diagnostic',json.dumps(diag),flush=True)
 del Xz,pz
 # Exact line-level old/new snapshot comparison. Only remove the new card from the new stream for this controlled comparison.
 if args.prior_snapshot:
  oldhash=sha(args.prior_snapshot);h=hashlib.sha256();counts=Counter();first_nonmatching=[];matched=0;extra=0
  with open(args.snapshot,'rb') as fn, open(args.prior_snapshot,'rb') as fo:
   head=next(fn);oldhead=next(fo);assert head==oldhead;h.update(head)
   file_index=head.decode().strip().split(',').index('file_name')
   assert file_index==0
   for i,line in enumerate(fn):
    name=line.split(b',',1)[0].decode();root=name.split('_y')[0];counts[root]+=1
    if root=='IMG_9429':extra+=1;continue
    h.update(line);ol=next(fo,None)
    if line==ol:matched+=1
    elif len(first_nonmatching)<10:first_nonmatching.append(i)
   trailing=next(fo,None)
  sn={'reference_rows':sum(counts.values()),'reference_cards':len(counts),'columns':len(head.decode().strip().split(',')),'added_card':'IMG_9429','added_rows':extra,'common_rows_byte_equal_in_order':matched,'first_nonmatching_indices':first_nonmatching,'prior_has_extra_trailing_row':trailing is not None,'filtered_reference_sha256':h.hexdigest(),'prior_sha256':oldhash,'filtered_reference_bytes_equal_prior':h.hexdigest()==oldhash,'historical_cause_of_extra_rows':'NOT_AUTHENTICATED; this is a comparison of present file bytes.'}
  dump(O/'SNAPSHOT_RECONCILIATION.json',sn);pd.DataFrame(sorted(counts.items()),columns=['card','rows']).to_csv(O/'snapshot_per_card.csv',index=False)
  assert sn['filtered_reference_bytes_equal_prior'] and sn['reference_rows']==363563 and extra==1892
  print('SNAPSHOT',json.dumps(sn),flush=True)
 # Read corrected snapshot and verify same internal-test scores; full sample profiles.
 print('LOAD full snapshot',flush=True);td=pd.read_csv(args.snapshot,low_memory=False)
 assert td.shape==(363563,109)
 t=pd.read_csv(args.model_dir/'tile_preds.csv');t['_order']=np.arange(len(t));K=['file_name','ann_id','scale'];assert not td.duplicated(K).any()
 j=t.merge(td,on=K,how='left',suffixes=('_saved',''),validate='one_to_one',indicator=True,sort=False).sort_values('_order');assert (j._merge=='both').all()
 fields={k:bool(np.array_equal(j[k+'_saved'].fillna(''),j[k].fillna(''))) for k in ['label','reviewed','review_tag','bbox_x','bbox_y','bbox_w','bbox_h']};assert all(fields.values())
 pi=bst.predict(xgb.DMatrix(impute(j[cols])))
 it={'rows':len(j),'all_keys_and_fields_match':fields,'same_float32_count':int(np.count_nonzero(pi==j.proba.to_numpy(np.float32))),'max_decimal_delta':float(np.max(np.abs(pi.astype(float)-j.proba.to_numpy()))),'decisions_equal_stored_threshold':bool(np.array_equal(pi>=obj['threshold'],j.pred.astype(bool))),'AP':float(average_precision_score(j.label,pi))}
 assert it['same_float32_count']==56843 and it['decisions_equal_stored_threshold'];dump(O/'INTERNAL_TEST_EXACT_SNAPSHOT.json',it)
 del j;gc.collect()
 rows=[]
 for i,c in enumerate(cols):
  a=td[c];b=df[c]
  rows.append(dict(feature=c,model_position=i,train_missing=int(a.isna().sum()),train_nonbinary=int((a.notna()&~a.isin([0,1])).sum()),train_min=a.min(),train_max=a.max(),train_unique=int(a.nunique()),audit_missing=int(b.isna().sum()),audit_nonbinary=int((b.notna()&~b.isin([0,1])).sum()),audit_min=b.min(),audit_max=b.max(),audit_unique=int(b.nunique()),model_median=imp.statistics_[i]))
 pd.DataFrame(rows).to_csv(O/'TRAIN_AUDIT_FEATURE_CONTRACT_VALUES.csv',index=False)
 gates_formula={'g_embed_equals_embed_sim':bool(np.array_equal(td.g_embed,td.embed_sim)), 'train_gate_nonbinary':{g:int((~td[g].isin([0,1])).sum()) for g in gates}}
 d=np.linalg.norm(td[[f'embed_pca_{k}' for k in range(32)]].to_numpy(np.float32),axis=1).astype(float)
 gates_formula['g_maha_max_abs_formula_delta']=float(np.max(np.abs(td.g_maha-np.clip(1-d/(d+5),0,1))))
 gates_formula['train_audit_cards_overlap']=sorted(set(td.file_name.str.extract(r'^(IMG_\d+)',expand=False))&set(df.img_folder));dump(O/'EXACT_SNAPSHOT_FORMULAS.json',gates_formula)
 pd.DataFrame([dict(feature=g,train_min=td[g].min(),train_max=td[g].max(),audit_missing=int(X[g].isna().sum()),imputed_value=float(imp.statistics_[cols.index(g)])) for g in gates]).to_csv(O/'GATE_INPUT_DISPOSITION.csv',index=False)
 # schema and all original byte identities rechecked at end
 assert all(sha(paths[k])==v for k,v in EXPECTED.items())
 dump(O/'IMPUTER_COMPATIBILITY.json',{'fallback_errors':imputation_fallbacks,'method':'finite fitted statistics substituted only at NaN cells; no fit or original state edits','environment':meta['environment']})
 dump(O/'RUN_STATUS.json',{'status':'COMPUTATIONAL_CHECKS_PASS','audit_rows':10097,'internal_rows':56843,'snapshot_rows':363563,'source_inputs_unchanged':True,'training':'NOT_RUN','vision_inference':'NOT_RUN','feature_values_reextracted':'NOT_RUN','task_closure':'PENDING_MANUSCRIPT_INTEGRATION_AND_REVIEW','fit_time_proof':'NOT_ESTABLISHED_BY_THIS_REPLAY'})
 print('COMPUTATIONAL_CHECKS_PASS',flush=True)
if __name__=='__main__':main()
