#!/usr/bin/env python3
"""Generate publication tables/figures from validated retained-score and S3 outputs.
No model loading, fitting, sampling or external data. All diagrams use source CSVs.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--metrics',type=Path,required=True);ap.add_argument('--s3',type=Path,required=True);ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise ValueError('Output must be new')
 a.out.mkdir(parents=True);figs=a.out/'figures';figs.mkdir();tables=a.out/'tables';tables.mkdir()
 valid=json.loads((a.metrics/'JR04_SAVED_SCORE_CHECK.json').read_text());svalid=json.loads((a.s3/'VERIFICATION_RESULT.json').read_text())
 if valid['status']!='SAVED_SCORE_AND_HISTORICAL_TABLE_CHECKS_PASS' or svalid['scientific_verification']!='PASS':raise ValueError('Required subchecks not PASS')
 data=pd.read_csv(a.metrics/'JR04_checkpoint_metrics_recomputed.csv');s2=pd.read_csv(a.metrics/'JR04_S2_recomputed.csv')
 r92=data[data['round'].eq(92)];raw=r92[r92['mode'].eq('raw')].iloc[0];prevalence=2032/10097
 pr=pd.read_csv(a.metrics/'JR04_r92_PR_coordinates.csv');roc=pd.read_csv(a.metrics/'JR04_r92_ROC_coordinates.csv')
 seven=pd.read_csv(a.metrics/'JR04_declared_seven_checkpoint_comparison.csv');s3=pd.read_csv(a.s3/'S3_RESULTS.csv').sort_values('threshold')
 def save(fig,name):
  fig.tight_layout()
  fig.savefig(figs/(name+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None,'Creator':'JR04 reproducible outputs v1'})
  fig.savefig(figs/(name+'.png'),dpi=220,bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(5.0,4.0));ax.step(pr.recall,pr.precision,where='post',label=f'r92: AP = {raw.pr_auc:.4f}')
 ax.axhline(prevalence,linestyle='--',label=f'CJ prevalence = {prevalence:.3f}')
 ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='Recall',ylabel='Precision');ax.legend(loc='lower left',fontsize=10)
 save(fig,'fig_pr_curve__xgb_recall')
 fig,ax=plt.subplots(figsize=(5.0,4.0));ax.plot(roc.fpr,roc.tpr,label=f'r92: ROC-AUC = {raw.roc_auc:.4f}')
 ax.plot([0,1],[0,1],linestyle='--',label='Chance reference');ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='False-positive rate',ylabel='True-positive rate');ax.legend(loc='lower right',fontsize=10)
 save(fig,'fig_roc_curve__xgb_recall')
 fig,ax=plt.subplots(figsize=(6.3,4.2));bars=ax.bar(['r'+str(x) for x in seven['round']],seven.pr_auc)
 ax.bar_label(bars,labels=[f'{v:.4f}' for v in seven.pr_auc],padding=3,fontsize=9)
 ax.set(ylim=(0,1.05),xlabel='Checkpoint',ylabel='Average precision (AP)');save(fig,'fig_bar_topk_pr_auc__xgb_recall')
 for mode,title in [('raw','Raw candidate counts'),('weighted','Review-weighted candidate mass')]:
  r=r92[r92['mode'].eq(mode)].iloc[0];mat=np.array([[r.tn,r.fp],[r.fn,r.tp]])
  cell=[[f"{(str(int(mat[i,j])) if mode=='raw' else f'{mat[i,j]:.1f}')}\n({100*mat[i,j]/mat[i].sum():.1f}%)" for j in range(2)] for i in range(2)]
  fig,ax=plt.subplots(figsize=(5.1,3.7));ax.axis('off')
  t=ax.table(cellText=cell,rowLabels=['non-CJ','CJ'],colLabels=['non-CJ','CJ'],loc='center',cellLoc='center',bbox=[.15,.23,.8,.63])
  t.auto_set_font_size(False);t.set_fontsize(12)
  for (i,j),c in t.get_celld().items():
   c.set_linewidth(.8)
   if i==0:c.set_text_props(weight='bold')
  ax.text(.55,.96,title,ha='center',va='top',fontsize=13,transform=ax.transAxes)
  ax.text(.55,.15,'Predicted class',ha='center',fontsize=11,transform=ax.transAxes)
  ax.text(-.09,.52,'True class',ha='center',va='center',rotation=90,fontsize=11,transform=ax.transAxes)
  ax.text(.55,.02,f'P = {r.precision:.3f}   R = {r.recall:.3f}   F1 = {r.f1:.3f}',ha='center',fontsize=10,transform=ax.transAxes)
  save(fig,'fig_confusion_'+mode+'__r92')
 hist=pd.read_csv(a.inputs/'learning_curve_table__xgb_recall.csv');complete=hist.n_missing_features_filled.eq(0)
 fig,ax=plt.subplots(figsize=(6.2,4.2));ax.scatter(hist.loc[complete,'round'],hist.loc[complete,'pr_auc'],s=17,label='Expected columns present')
 ax.scatter(hist.loc[~complete,'round'],hist.loc[~complete,'pr_auc'],marker='x',s=34,label='Missing columns: diagnostic')
 ax.set(xlabel='Archived round',ylabel='Candidate-audit average precision (AP)')
 rr=hist[hist['round'].eq(92)].iloc[0];ax.annotate('r92',xy=(92,rr.pr_auc),xytext=(7,5),textcoords='offset points',fontsize=10)
 ax.legend(loc='lower right',fontsize=8);ax.margins(y=.14);save(fig,'fig_learning_curve__xgb_recall')
 fig,ax=plt.subplots(figsize=(6.2,4.4));s=s3.sort_values('mean_pre_nms_candidates_per_card')
 ax.plot(s.mean_pre_nms_candidates_per_card,s.recall,marker='o')
 for _,r in s.iterrows():ax.annotate(f"$\\tau={r.threshold:.1f}$",(r.mean_pre_nms_candidates_per_card,r.recall),xytext=(7,5),textcoords='offset points',fontsize=10)
 ax.set(xlabel='Mean score-eligible candidates per card (pre-NMS)',ylabel='Micro point-coverage recall',xlim=(250,351),ylim=(.725,.93))
 ax.grid(True,alpha=.25);save(fig,'fig_harmonized_threshold_sweep__plot')
 def write_rows(name,rows): (tables/name).write_text(''.join(' & '.join(row)+r' \\'+'\n' for row in rows))
 rawall=data[data['mode'].eq('raw')].sort_values('pr_auc',ascending=False)
 write_rows('Table5_rows.tex',[[str(int(r['round']))+(r'$^{\dagger}$' if r['round'] in [74,76,82] else '')]+[f'{r[k]:.4f}' for k in ['pr_auc','roc_auc','precision','recall','f1']] for _,r in rawall.iterrows()])
 names={'border-or-tiny':'Border-or-tiny','non-border/tiny':'Non-border/tiny','overall':'Overall'}
 write_rows('TableS2_rows.tex',[[names[r['slice']],str(int(r.n)),str(int(r.positives))]+[str(int(r[k])) for k in ['tp','fp','fn']]+[f'{r[k]:.4f}' for k in ['pr_auc','precision','recall','f1']] for _,r in s2[s2['mode'].eq('raw')].iterrows()])
 write_rows('TableS5_rows.tex',[[('Raw' if r['mode']=='raw' else 'Review-weighted')]+[(str(int(r[k])) if r['mode']=='raw' else f'{r[k]:.1f}') for k in ['tn','fp','fn','tp']]+[f'{r[k]:.4f}' for k in ['pr_auc','precision','recall','f1']] for _,r in r92.iterrows()])
 write_rows('TableS3_rows.tex',[[f'{r.threshold:.1f}',f'{r.recall:.3f}',f'{int(r.covered_points)}/{int(r.gt_points)}',f'{r.mean_pre_nms_candidates_per_card:.1f}'] for _,r in s3.iterrows()])
 r92.to_csv(tables/'TableS5_full_precision.csv',index=False);rawall.to_csv(tables/'Table5_full_precision.csv',index=False);s2.to_csv(tables/'TableS2_full_precision.csv',index=False);s3.to_csv(tables/'TableS3_full_precision.csv',index=False)
 hist.to_csv(tables/'Figure10_122_record_data.csv',index=False)
 sources=[a.metrics/n for n in ['JR04_checkpoint_metrics_recomputed.csv','JR04_S2_recomputed.csv','JR04_r92_PR_coordinates.csv','JR04_r92_ROC_coordinates.csv','JR04_declared_seven_checkpoint_comparison.csv']]+[a.s3/'S3_RESULTS.csv',a.inputs/'learning_curve_table__xgb_recall.csv']
 (a.out/'OUTPUT_BINDING.json').write_text(json.dumps({'generator_sha256':sha(__file__),'input_sha256':{str(p):sha(p) for p in sources},'outputs':{str(p.relative_to(a.out)):sha(p) for p in sorted(a.out.rglob('*')) if p.is_file()},'role':'Generated tables and figures; source values unchanged except authorized S3 reevaluation.'},indent=2)+'\n')
 print('GENERATED_FROM_VALIDATED_RESULTS')
if __name__=='__main__':main()
