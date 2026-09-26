#!/usr/bin/env python3
"""Refresh 57 reviewer-response locators after the JR04 manuscript build.
Uses the supplied JR03 anchor map and verifies every resulting line endpoint in PDF.
"""
import argparse,re,json,hashlib
from pathlib import Path
import fitz

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--aux',type=Path,required=True);ap.add_argument('--pdf',type=Path,required=True);ap.add_argument('--baseline-map',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 old=json.loads(a.baseline_map.read_text());aux=a.aux.read_text();doc=fitz.open(a.pdf)
 ln={m[1]:{'line':int(m[2])+1,'page':int(m[3])} for m in re.finditer(r'\\newlabel\{((?:revloc|jr0[1-4]):[^}]+)\}\{\{\\getpagewiselinenumber \{(\d+)\}\}\{(\d+)\}',aux)}
 floats={m[1]:{'number':m[2],'page':int(m[3]),'type':m[1].split(':')[0]} for m in re.finditer(r'\\newlabel\{((?:fig|tab):[^}]+)\}\{\{([^{}]+)\}\{(\d+)\}',aux)}
 margins=[];texts=[]
 for p in doc:
  nums=set();texts.append(p.get_text())
  for b in p.get_text('dict')['blocks']:
   for l in b.get('lines',[]):
    for sp in l['spans']:
     if sp['size']<6 and (sp['bbox'][0]<40 or sp['bbox'][0]>560) and sp['text'].strip().isdigit():nums.add(int(sp['text'].strip()))
  margins.append(nums)
 for k,v in ln.items():
  if v['line'] not in margins[v['page']-1]:raise ValueError(f'Endpoint not rendered: {k} {v}')
 def fmt(s,e):
  x,y=ln[s],ln[e];return (f"p.~{x['page']}" if x['page']==y['page'] else f"pp.~{x['page']}--{y['page']}")+f", lines {x['line']}--{y['line']}"
 def ploc(prefix):return fmt(prefix+':start',prefix+':end')
 def obj(label,title):return title+f" (p.~{floats[label]['page']})"
 rev=old['line_label_map'];refs=[i+1 for i,t in enumerate(texts) if i>20 and re.search(r'\bReferences\b',t)]
 refstart=refs[0];refend=len(doc)
 ranged_changes=[]
 def replace_range(m):
  p1=int(m[1]);p2=int(m[2] or m[1]);l1=int(m[3]);l2=int(m[4])
  starts=[k for k,v in rev.items() if v=={'line':l1,'page':p1}]
  ends=[k for k,v in rev.items() if v=={'line':l2,'page':p2}]
  if starts and ends:new=fmt(starts[0],ends[0])
  elif l1>1800 and p1>=30:
   new=f'pp.~{refstart}--{refend}, lines {min(margins[refstart-1])}--{max(margins[refend-1])}'
  else:raise ValueError('Unmapped old range '+m[0])
  ranged_changes.append({'before':m[0],'after':new});return new
 bynumber={(v['type'],v['number']):v['page'] for v in floats.values()}
 def replace_float(m):
  typ='tab' if m[2]=='Table' else 'fig';key=(typ,m[3]);
  if key not in bynumber:raise ValueError('Unknown float '+str(key))
  return m[1]+f' (p.~{bynumber[key]})'
 p=a.source/'response/response_to_reviewers.tex';s=p.read_text();records=[]
 starts=list(re.finditer(r'\\subsection\*\{([^:}]+):',s));pieces=[];last=0
 for n,m in enumerate(starts):
  end=starts[n+1].start() if n+1<len(starts) else len(s);block=s[m.start():end]
  lm=re.search(r'\\location\{([^\n]+)\}',block)
  if not lm:raise ValueError('Missing location '+m[1])
  loc=lm[1]
  loc=re.sub(r'p{1,2}\.~(\d+)(?:--(\d+))?, lines (\d+)--(\d+)',replace_range,loc)
  loc=re.sub(r'((Table|Fig\.)~(S?\d+[a-z]?)(?: \([^\n)]*\))?) \(p\.~\d+\)',replace_float,loc)
  loc=re.sub(r'(Table~S1 \(pp\.~)\d+--\d+(\))',lambda m:m[1]+str(floats['tab:supp_config_repro']['page'])+'--'+str(floats['tab:supp_stratified_border_tiny']['page'])+m[2],loc)
  loc=re.sub(r'p\.~\d+, references \[15\]--\[18\]',f'p.~{refstart}, references [15]--[18]',loc)
  rid=m[1]
  if rid in ['R2-Q2-C3','R2-Q3-C3','R2-Q3-C5']:
   loc='; '.join(['Methods, common candidate-score reproduction ('+ploc('jr04:metrics')+')',obj('tab:perf_summary','Table~5'),obj('fig:pr_roc','Fig.~5'),obj('fig:topk','Fig.~6'),obj('fig:confmat','Fig.~7'),obj('tab:supp_stratified_border_tiny','Table~S2'),obj('tab:supp_raw_weighted_r92','Table~S5'),obj('tab:supp_feature_contract','Table~S8')])+'.'
  if rid in ['R1-MIN-5','R2-Q4-C3']:
   loc='; '.join(['Methods, versioned point sensitivity ('+ploc('jr04:sweep')+')','Results, revised point sweep ('+ploc('jr04:s3results')+')',obj('tab:supp_threshold_sweep','Table~S3'),obj('fig:supp_threshold_sweep','Fig.~S1'),obj('tab:supp_instance_threshold','Table~S4'),obj('tab:instance_mask_eval','Table~8')])+'.'
  if rid=='R2-Q3-C1':loc+=' '+obj('fig:pr_roc','Fig.~5')+'.'
  loc=loc.replace('.;',';').replace('..','.')
  newblock=block[:lm.start(1)]+loc+block[lm.end(1):]
  pieces += [s[last:m.start()],newblock];last=end
  records.append({'comment_id':rid,'location_tex':loc})
 pieces.append(s[last:]);new=''.join(pieces)
 assert len(records)==57
 p.write_text(new)
 a.out.write_text(json.dumps({'line_label_map':ln,'float_labels':floats,'rendered_endpoint_count':len(ln),'all_line_endpoints_present':True,'all57_locations':records,'range_changes':ranged_changes,'manuscript_sha256':hashlib.sha256(a.pdf.read_bytes()).hexdigest()},indent=2)+'\n')
 print('Refreshed',len(records),'replies;',len(ln),'line endpoints confirmed')
if __name__=='__main__':main()
