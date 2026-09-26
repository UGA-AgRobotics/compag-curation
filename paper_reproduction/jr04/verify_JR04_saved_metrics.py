#!/usr/bin/env python3
"""JR04 bounded verification of retained scores and plotting provenance.
No pickle/model loading, fitting, notebook execution, or point-sweep replacement.
Inputs are hash-pinned; output must be a fresh directory. Figures are DRAFT.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve

PINS = {
'checkpoint_scores_all_rounds.csv': '5fa304558685bb8a5ab6d056b83e05277c3ed5139eced417afd5a5e32317102a',
'learning_curve_table__xgb_recall.csv': 'f016a1c3292353444679dfbac2c35bfb2149cb2eeabb5a7c19e219e37ab59f6e',
'model_metrics__xgb_recall.csv': 'b95e0b83869b09d689e92a5fef17f420c74d06d00732bbcd4b2e5e4030409f35',
'model_scoring_diagnostics.tsv': '99ed8045dc2570a7166cea0e0205a5d309172ddd636d216aae8c4ab185d00e4b',
'scored_candidates_r92.csv': 'b27eee34994efb5af61cd618b27712dd5120e363d4606019a7b0daabebe7a55c',
'supp_table_s2_reconciled_raw_and_weighted.csv': '3c24c7dd9e73295904c7d9bdafd26b2c7f56307bebded9fbf6341a37421fc18a',
'table4_reconciled_raw_and_weighted.csv': 'e2447a22ff85e54d99cd9e376c56f4c953d9d2f04bdbd55f17aac0361421fcb2',
}

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict:
    require(len(y) == len(p) == len(w) and len(y) > 0, 'Empty or unequal input arrays')
    require(np.isin(y, [0, 1]).all() and np.isfinite(p).all() and ((0<=p)&(p<=1)).all(), 'Bad labels/scores')
    require(np.isfinite(w).all() and (w > 0).all(), 'Invalid weights')
    pred = p >= .5
    tn, fp, fn, tp = [float(w[m].sum()) for m in [
        (y==0)&~pred, (y==0)&pred, (y==1)&~pred, (y==1)&pred]]
    div=lambda a,b: a/b if b else 0.
    pr=div(tp,tp+fp); rc=div(tp,tp+fn)
    prec,rec,thr=precision_recall_curve(y,p,sample_weight=w)
    f1=2*prec*rec/np.clip(prec+rec,1e-12,None)
    best=int(np.argmax(f1[:-1]))
    return dict(tn=tn,fp=fp,fn=fn,tp=tp,precision=pr,recall=rc,
        f1=div(2*pr*rc,pr+rc),specificity=div(tn,tn+fp),fpr=div(fp,tn+fp),
        fnr=div(fn,tp+fn),accuracy=div(tp+tn,w.sum()),
        pr_auc=float(average_precision_score(y,p,sample_weight=w)),
        roc_auc=float(roc_auc_score(y,p,sample_weight=w)),
        best_threshold=float(thr[best]),best_precision=float(prec[best]),
        best_recall=float(rec[best]),best_f1=float(f1[best]))

def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--figures',action='store_true')
    args=ap.parse_args()
    require(not args.out.exists(),'Refusing to overwrite output directory')
    for name,pin in PINS.items():
        f=args.inputs/name
        require(f.is_file(),f'Missing input: {name}')
        require(hashlib.sha256(f.read_bytes()).hexdigest()==pin,f'Input identity mismatch: {name}')
    args.out.mkdir(parents=True)
    sc=pd.read_csv(args.inputs/'checkpoint_scores_all_rounds.csv')
    r92=pd.read_csv(args.inputs/'scored_candidates_r92.csv')
    diag=pd.read_csv(args.inputs/'model_scoring_diagnostics.tsv',sep='\t',keep_default_na=False)
    table=pd.read_csv(args.inputs/'table4_reconciled_raw_and_weighted.csv')
    s2=pd.read_csv(args.inputs/'supp_table_s2_reconciled_raw_and_weighted.csv')
    hist=pd.read_csv(args.inputs/'model_metrics__xgb_recall.csv',keep_default_na=False)
    learn=pd.read_csv(args.inputs/'learning_curve_table__xgb_recall.csv',keep_default_na=False)
    keys=['img_folder','image','id']
    require(len(sc)==len(r92)==10097 and not sc.duplicated(keys).any(),'Bad canonical population/IDs')
    for col in keys+['human_label','action','review_weight']:
        require(sc[col].equals(r92[col]),f'Canonical row identity/order mismatch: {col}')
    require(np.array_equal(sc.score_r92.to_numpy(),r92.canonical_score.to_numpy()),'r92 score projections differ')
    require(np.array_equal((r92.canonical_score.to_numpy()>=.5).astype(int),r92.canonical_prediction.to_numpy()),'Decisions differ')
    require(int(sc.human_label.sum())==2032 and sc.img_folder.nunique()==10,'Wrong labels/cards')
    weight_map={'accept':1.,'flip':1.,'sus_accept':.4,'sus_flip':.4}
    require(sc.action.isin(weight_map).all(),'Unexpected included action')
    require(np.array_equal(sc.action.map(weight_map).to_numpy(),sc.review_weight.to_numpy()),'Weights differ from declared action map')
    y=sc.human_label.to_numpy(int); w=sc.review_weight.to_numpy(float)
    rounds=sorted(int(c[7:]) for c in sc if re.fullmatch(r'score_r\d+',c))
    require(set(rounds)==set(table['round'])==set(diag['round']),'Checkpoint population mismatch')
    comparisons=[]; rows=[]
    def compare(scope, mode, computed, saved):
        for key,val in computed.items():
            actual=float(saved[f'{mode}_{key}'])
            delta=abs(actual-val)
            require(delta<=1e-10,f'{scope}/{mode}/{key} differs by {delta}')
            comparisons.append(dict(scope=scope,mode=mode,metric=key,recomputed=val,saved=actual,abs_delta=delta))
    for rn in rounds:
        p=sc[f'score_r{rn}'].to_numpy(float)
        saved=table.loc[table['round']==rn].iloc[0]
        missing=int(diag.loc[diag['round']==rn,'n_missing_features'].iloc[0])
        for mode,ww in [('raw',np.ones(len(y))),('weighted',w)]:
            m=metrics(y,p,ww); compare(f'r{rn}',mode,m,saved)
            rows.append(dict(round=rn,mode=mode,n=len(y),threshold=.5,missing_column_names=missing,**m))
    pd.DataFrame(rows).to_csv(args.out/'JR04_checkpoint_metrics_recomputed.csv',index=False)
    flags=r92.is_border_or_tiny.to_numpy(bool)
    require(flags.sum()==8056 and (~flags).sum()==2041,'Unexpected slice population')
    srows=[]
    for sl,mask in [('border-or-tiny',flags),('non-border/tiny',~flags),('overall',np.ones(len(y),bool))]:
        saved=s2.loc[s2['slice'].str.lower()==sl].iloc[0]
        for mode,ww in [('raw',np.ones(mask.sum())),('weighted',w[mask])]:
            m=metrics(y[mask],sc.score_r92.to_numpy(float)[mask],ww); compare(sl,mode,m,saved)
            srows.append(dict(slice=sl,mode=mode,n=int(mask.sum()),positives=int(y[mask].sum()),**m))
    pd.DataFrame(srows).to_csv(args.out/'JR04_S2_recomputed.csv',index=False)
    pd.DataFrame(comparisons).to_csv(args.out/'JR04_metric_comparisons.csv',index=False)
    require(len(hist)==134 and hist.model_rel.nunique()==134,'Unexpected historical metrics table')
    require((hist.n_test==10097).all() and (hist.fixed_thr==.5).all(),'Historical table population/threshold differs')
    history_cmp=[]
    mappings={'pr_auc':'raw_pr_auc','roc_auc':'raw_roc_auc','f1_fixed':'raw_f1','prec_fixed':'raw_precision','rec_fixed':'raw_recall'}
    for _, row in table.iterrows():
        rel=Path(row.model_path).parent.name+'/'+Path(row.model_path).name
        match=hist.loc[hist.model_rel==rel]
        require(len(match)==1,f'No unique historical row for {rel}')
        for old,new in mappings.items():
            delta=abs(float(match.iloc[0][old])-float(row[new])); require(delta<=1e-12,'Historical metric conflict')
            history_cmp.append(dict(model_rel=rel,metric=old,abs_delta=delta))
    pd.DataFrame(history_cmp).to_csv(args.out/'JR04_historical_vs_canonical.csv',index=False)
    selected=(hist.loc[hist.model_rel.str.contains('_hybrid',case=False,regex=False)]
              .sort_values(['round','pr_auc'],ascending=[True,False])
              .groupby('round',as_index=False).first().sort_values('round').reset_index(drop=True))
    require(len(learn)==len(selected)==122,'Unexpected historical per-round count')
    require(learn.model_rel.equals(selected.model_rel) and learn['round'].equals(selected['round']), 'Historical round member selection differs')
    for col in ['pr_auc','roc_auc','f1_fixed','n_features_schema','n_missing_features_filled']:
        require(np.array_equal(learn[col].to_numpy(),selected[col].to_numpy()),f'Historical round values differ: {col}')
    hist.loc[hist.n_missing_features_filled>0].to_csv(args.out/'JR04_historical_input_incomplete_models.csv',index=False)
    eligible=sorted(diag.loc[diag.n_missing_features==0,'round'].astype(int).tolist())
    require(eligible==[53,73,92,106,110,112,121],'Unexpected column-name-available subset')
    raw=pd.DataFrame(rows).query("mode=='raw'")
    chosen=raw.loc[raw['round'].isin(eligible)].sort_values('pr_auc',ascending=False)
    chosen.to_csv(args.out/'JR04_declared_seven_checkpoint_comparison.csv',index=False)
    p=sc.score_r92.to_numpy(float)
    prec,rec,thr=precision_recall_curve(y,p)
    pd.DataFrame({'recall':rec,'precision':prec}).to_csv(args.out/'JR04_r92_PR_coordinates.csv',index=False)
    fpr,tpr,rt=roc_curve(y,p)
    pd.DataFrame({'fpr':fpr,'tpr':tpr,'threshold':rt}).to_csv(args.out/'JR04_r92_ROC_coordinates.csv',index=False)
    summary=dict(status='SAVED_SCORE_AND_HISTORICAL_TABLE_CHECKS_PASS',task_status='SUBCHECK_ONLY',
        models_scored_or_trained=False,notebook_executed=False,point_evaluator_executed=False,
        canonical_rows=len(sc),cards=sc.img_folder.nunique(),positive_rows=int(y.sum()),
        metric_comparisons=len(comparisons),max_abs_delta=max(c['abs_delta'] for c in comparisons),
        historical_metric_comparisons=len(history_cmp),historical_models=len(hist),
        historical_input_incomplete_models=int((hist.n_missing_features_filled>0).sum()),
        historical_per_round_rows=len(learn),historical_per_round_values_verified=True,
        eligible_column_name_subset=eligible,
        pr_roc_historical_actual_curve_membership='NOT_RECONSTRUCTED_FROM_AGGREGATE_TABLE',
        s3='SEPARATE_RUNNER: run_s3_v1.py',
        manuscript_modified=False,figures='DRAFT_ONLY' if args.figures else 'NOT_GENERATED',
        input_sha256=PINS,
        scope='Recomputed saved candidate metrics and table transformations; no proof of historical training independence or measured completeness of all named inputs.')
    (args.out/'JR04_SAVED_SCORE_CHECK.json').write_text(json.dumps(summary,indent=2)+'\n')
    if args.figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figs=args.out/'draft_figures'; figs.mkdir()
        def save(fig,name):
            fig.tight_layout();fig.savefig(figs/(name+'.png'),dpi=220,bbox_inches='tight');plt.close(fig)
        r=raw.loc[raw['round']==92].iloc[0]
        fig,ax=plt.subplots(figsize=(5.4,4.3))
        ax.step(rec,prec,where='post',label=f'r92: AP={r.pr_auc:.4f}')
        ax.axhline(y.mean(),linestyle='--',label=f'CJ prevalence={y.mean():.3f}')
        ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='Recall',ylabel='Precision',title='Retrospective candidate audit | r92')
        ax.legend(loc='lower left');save(fig,'DRAFT_r92_PR')
        fig,ax=plt.subplots(figsize=(5.4,4.3))
        ax.plot(fpr,tpr,label=f'r92: ROC-AUC={r.roc_auc:.4f}')
        ax.plot([0,1],[0,1],linestyle='--',label='Chance reference')
        ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='False-positive rate',ylabel='True-positive rate',title='Retrospective candidate audit | r92')
        ax.legend(loc='lower right');save(fig,'DRAFT_r92_ROC')
        fig,ax=plt.subplots(figsize=(6.8,4.3))
        bars=ax.bar(['r'+str(v) for v in chosen['round']],chosen.pr_auc)
        ax.bar_label(bars,labels=[f'{v:.4f}' for v in chosen.pr_auc],padding=3,fontsize=9)
        ax.set(ylim=(0,1.05),ylabel='Average precision (AP)',xlabel='Checkpoint',
               title='Seven named checkpoints | same retained audit scores')
        save(fig,'DRAFT_seven_checkpoint_AP')
    print(summary['status']);print('Numeric checks:',len(comparisons),'historical comparisons:',len(history_cmp))

if __name__=='__main__':
    main()
