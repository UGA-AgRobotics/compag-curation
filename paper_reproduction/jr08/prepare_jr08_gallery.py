#!/usr/bin/env python3
"""JR08: preserve author free-text comments; audit saved associations; relayout unchanged images.
No model/evaluator/training code is loaded or executed. Requires Pillow, reportlab and PyMuPDF.
Original ZIPs and author JSON are read only; --out must not exist.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, io, json, re, sys, zipfile
from pathlib import Path
from PIL import Image

CORE_SHA='cd03ded8f4f7ac0309f1d5fb89f7454569d4f0798414f37d0143594c3feb70a4'
GALLERY_SHA='74c1408e7f27e6eab943adeab03d7810acebac1831ec30bb4cfee4a613dbed91'
AUDIT_PREFIX='JR06_Verified_Evidence_Pending_Scope/inputs/core/mask_study/results/run_20260803_184228_8PwhfM/'
EFFORT_PREFIX='JR06_Verified_Evidence_Pending_Scope/inputs/core/effort/'

def sha(b:bytes)->str: return hashlib.sha256(b).hexdigest()
def csvrows(b):return list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))
def write_json(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def write_csv(p,rs):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]),lineterminator='\n');w.writeheader();w.writerows(rs)
def primary(rs):return [r for r in rs if float(r['score_threshold'])==.5 and float(r['mask_iou_threshold'])==.5]

def font_paths():
    choices=[Path('/usr/share/fonts/truetype/liberation'),Path('/usr/share/fonts/truetype/liberation2')]
    for p in choices:
        if (p/'LiberationSans-Regular.ttf').exists():return p/'LiberationSans-Regular.ttf',p/'LiberationSans-Bold.ttf'
    raise RuntimeError('Liberation Sans fonts are required for deterministic publication rendering.')

def render_gallery(core,main,authors,out):
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.utils import ImageReader
    import fitz
    reg,bold=font_paths();pdfmetrics.registerFont(TTFont('JR08Sans',str(reg)));pdfmetrics.registerFont(TTFont('JR08SansB',str(bold)))
    W,H=518.4,604.8
    path=out/'JR08_Failure_Gallery_Author_Reviewed.pdf'
    c=canvas.Canvas(str(path),pagesize=(W,H),invariant=1,pageCompression=1)
    c.setTitle('Author-reviewed examples of saved matching and topology outcomes')
    c.setAuthor('Hasan Jahanifar; publication layout assembled from retained evidence')
    c.setSubject('Twelve unchanged photograph/overlay pairs; author free-text observations; no new predictions')
    def txt(x,y,s,size=8,font='JR08Sans'):
        if pdfmetrics.stringWidth(s,font,size)>W-2*x and x<10:raise ValueError('Header too wide')
        c.setFillColorRGB(0,0,0);c.setFont(font,size);c.drawString(x,y,s)
    txt(7,H-15,'Author-reviewed examples of saved matching and topology outcomes',10.6,'JR08SansB')
    txt(7,H-29,'FP / FN: unmatched prediction / reference; duplicate, split and merge: saved events.',8)
    swatches=[(7,(0,114/255,178/255),'Reference (GT)'),(143,(204/255,121/255,167/255),'Event prediction(s)'),(315,(230/255,159/255,0),'Linked prediction')]
    for x,col,label in swatches:
        c.setFillColorRGB(*col);c.rect(x,H-43,7,7,fill=1,stroke=0);txt(x+11,H-42,label,8.5)
    txt(7,H-56,'Raw image on the left; saved overlay on the right. Coincident outlines can hide one another.',8)
    rowtop=H-71;pw=(W-14-12)/3;rh=128
    pixelrows=[]
    for i,r in enumerate(main):
        letter=chr(97+i);a=authors[letter];x=7+(i%3)*(pw+6);top=rowtop-(i//3)*rh
        title=f"({letter}) {r['outcome_event_label']} | {r['card_id'][4:]}"
        assert pdfmetrics.stringWidth(title,'JR08SansB',8.35)<pw
        txt(x,top,title,8.35,'JR08SansB')
        desc=a['short_description_en']
        if letter in 'dl':desc='No overlapping final prediction'
        if letter in 'bc':desc='Partial mask near tile edge*'
        assert pdfmetrics.stringWidth(desc,'JR08Sans',7.85)<pw
        txt(x,top-11,desc,7.85)
        txt(x,top-21,'Raw',7.3);txt(x+pw/2,top-21,'Saved overlay',7.3)
        name=f"{int(r['inspection_order_zero_based']):03d}_{r['candidate_id']}.png"
        im=Image.open(io.BytesIO(core.read('evidence/inspection_candidates/'+name))).convert('RGB')
        pair=im.crop((10,80,1450,800));ph=pw/2
        c.drawImage(ImageReader(pair),x,top-26-ph,pw,ph,mask=None)
        pixelrows.append({'panel':letter,'candidate_id':r['candidate_id'],'source_png':name,'source_png_sha256':sha(core.read('evidence/inspection_candidates/'+name)),'display_crop_xyxy':'10,80,1450,800','display_rgb_sha256':sha(pair.tobytes()),'width_px':pair.width,'height_px':pair.height})
    txt(7,17,'*Near-edge geometry supports possible clipping, not a proven cause. Panels (c) and (g) share one GT.',7.9)
    txt(7,6,'Descriptions use author observations and saved records; unaddressed archival appearance tags are not endorsed.',7.5)
    c.showPage();c.save()
    doc=fitz.open(path);ims=doc[0].get_images(full=True);assert len(ims)==12
    embedded=[]
    for im in ims:
        pix=fitz.Pixmap(doc,im[0]); assert pix.n==3;embedded.append(sha(pix.samples))
    assert sorted(embedded)==sorted(r['display_rgb_sha256'] for r in pixelrows)
    for r in pixelrows:r['exact_rgb_present_in_pdf']=r['display_rgb_sha256'] in embedded
    bounds=[]
    for block in doc[0].get_text('dict')['blocks']:
        if 'lines' in block:
            for line in block['lines']:
                for span in line['spans']:
                    b=span['bbox'];assert b[0]>=0 and b[1]>=0 and b[2]<=W+.1 and b[3]<=H+.1
                    bounds.append({'text':span['text'],'bbox':list(b)})
    doc.close();write_csv(out/'JR08_Gallery_Pixel_Integrity.csv',pixelrows)
    write_json(out/'JR08_Gallery_Render_Check.json',{'pages':1,'image_objects':12,'unchanged_rgb_pairs':12,'text_bounds_pass':True,'font_sha256':{'regular':sha(reg.read_bytes()),'bold':sha(bold.read_bytes())},'pdf_sha256':sha(path.read_bytes()),'text_spans':bounds})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--core-zip',type=Path,required=True);p.add_argument('--jr06-zip',type=Path,required=True);p.add_argument('--author-record',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():p.error('--out exists; no files overwritten')
    before={str(x):sha(x.read_bytes()) for x in [a.core_zip,a.jr06_zip,a.author_record]}
    assert before[str(a.core_zip)]==CORE_SHA
    author=json.loads(a.author_record.read_text());verbatim=a.author_record.parent/author['verbatim_file'];assert sha(verbatim.read_bytes())==author['verbatim_sha256']
    assert len(author['panels'])==12
    authors={r['panel']:r for r in author['panels']};assert list(authors)==list('abcdefghijkl')
    z=zipfile.ZipFile(a.core_zip);z6=zipfile.ZipFile(a.jr06_zip);assert z.testzip() is None and z6.testzip() is None
    assert sha(z.read('render/fig_failure_case_gallery_manuscript.pdf'))==GALLERY_SHA
    manifest=json.loads(z.read('COLLECTION_MANIFEST.json'));copied=[r for r in manifest['files'] if r['status']=='COPIED']
    for r in copied:assert sha(z.read(r['archive_member']))==r['sha256'] and len(z.read(r['archive_member']))==r['bytes']
    sm=csvrows(z.read('evidence/source_manifest_sha256.csv'));roles={r['source_role']:r for r in sm}
    def bound(n,role):
        b=z6.read(n);assert sha(b)==roles[role]['sha256_before']==roles[role]['sha256_after'];return b
    pe=primary(csvrows(bound(AUDIT_PREFIX+'prediction_errors.csv','polygon_prediction_errors')))
    ge=primary(csvrows(bound(AUDIT_PREFIX+'ground_truth_errors.csv','polygon_ground_truth_errors')))
    mp=primary(csvrows(bound(AUDIT_PREFIX+'matched_pairs.csv','polygon_matched_pairs')))
    te=primary(csvrows(bound(AUDIT_PREFIX+'topology_events.csv','polygon_topology_events')))
    coco=json.loads(bound(AUDIT_PREFIX+'frozen_inputs/instances_default.json','polygon_ground_truth_coco'));gt={f"gt:{r['id']}":r for r in coco['annotations']}
    mainrows=[r for r in csvrows(z.read('evidence/failure_case_panel_manifest.csv')) if r['document']=='MAIN'];assert len(mainrows)==12
    pred={};tiles={}
    for card in ['IMG_9459','IMG_9487']:
        prefix=EFFORT_PREFIX+card[4:]+'/'
        for r in csvrows(bound(prefix+'detections.csv','saved_predictions:'+card)):pred[r['image']+':'+r['id']]=r
        for r in csvrows(bound(prefix+'tiles_index.csv','saved_tile_index:'+card)):tiles[r['tile_name']]=r
    detail=[];borders=[];associations=[]
    for i,r in enumerate(mainrows):
        letter=chr(97+i);ar=authors[letter];assert r['candidate_id']==ar['candidate_id']
        name=f"{int(r['inspection_order_zero_based']):03d}_{r['candidate_id']}.json";md=json.loads(z.read('evidence/inspection_candidates/'+name));ents=md['entities']
        gids=[e['entity_id'] for e in ents if e['entity_type']=='ground_truth'];pids=[e['entity_id'] for e in ents if e['entity_type']=='prediction']
        evidence={}
        if letter in 'bc':
            pr=pred[pids[0]];tile=tiles[pr['image']];poly=ast.literal_eval(pr['poly']);gpoints=[(s[j],s[j+1]) for s in gt[gids[0]]['segmentation'] for j in range(0,len(s),2)]
            x,y,w,h=[int(tile[k]) for k in ['x','y','crop_w','crop_h']]
            mn,mx=min(poly[1::2]),max(poly[1::2]);gymin,gymax=min(p[1] for p in gpoints),max(p[1] for p in gpoints)
            edge='top' if letter=='b' else 'bottom';by=y if edge=='top' else y+h
            near=(mn<=1 if edge=='top' else mx>=h-1);cross=gymin<by<gymax
            assert near and cross
            evidence={'edge':edge,'tile_edge_full_y':by,'prediction_local_y_min':mn,'prediction_local_y_max':mx,'gt_full_y_min':gymin,'gt_full_y_max':gymax,'reference_crosses_original_tile_edge':cross,'prediction_reaches_edge_neighborhood':near,'conclusion':'Compatible with partial tile-edge coverage; no counterfactual causal test performed'}
            borders.append({'panel':letter,'prediction_key':pids[0],'gt_key':gids[0],**evidence})
        if letter in 'dl':
            q=next(v for v in ge if v['ground_truth_key']==gids[0]);assert q['best_prediction_key']=='' and float(q['best_mask_iou'])==0 and pids==[]
            evidence={'best_saved_mask_iou':0,'saved_linked_prediction':None,'drawn_prediction_count':0,'meaning':'No positive-overlap retained final prediction in this saved record, not absence of all upstream proposals'}
        if letter in 'ef':
            q=next(v for v in pe if v['prediction_key']==pids[0]);m=next(v for v in mp if v['ground_truth_key']==gids[0]);assert q['error_type']=='DUPLICATE' and float(q['best_mask_iou'])>=.5 and m['prediction_key']==pids[1]
            evidence={'duplicate_iou':float(q['best_mask_iou']),'representative_iou':float(m['iou']),'matching_unchanged':True,'author_neighbor_label':'author interpretation, not new biological validation' if letter=='e' else None}
        if letter in 'ghijk':
            matches=[t for t in te if t['ground_truth_keys'].split(';')==gids and t['prediction_keys'].split(';')==pids];assert len(matches)==1;evidence=matches[0]
            assert (len(gids),len(pids))==((2,1) if letter in 'ij' else (1,2))
        associations.append({'panel':letter,'candidate_id':r['candidate_id'],'ground_truth_ids':gids,'prediction_ids':pids,'saved_outcome':r['outcome_event_label'],'evidence':evidence})
        detail.append({'panel':letter,'candidate_id':r['candidate_id'],'card':r['card_id'],'saved_outcome':r['outcome_event_label'],'author_translation_en':ar['author_translation_en'],'author_review_status':ar['review_status'],'F_glare':ar['F_glare'],'G_blur':ar['G_blur'],'H_ink':ar['H_ink'],'I_confounder':ar['I_confounder'],'J_small_border':ar['J_small_border'],'publication_description':ar['short_description_en'],'disposition':ar['disposition'],'original_present_tags':r['visual_tags_confirmed_json'],'original_uncertain_tags':r['visual_tags_uncertain_json'],'unmentioned_tags':'NOT_ASSESSED','gt_ids':';'.join(gids),'prediction_ids':';'.join(pids)})
    assert associations[2]['ground_truth_ids']==associations[6]['ground_truth_ids']==['gt:6']
    a.out.mkdir(parents=True)
    write_csv(a.out/'JR08_Author_Comments_and_Dispositions_EN.csv',detail);write_csv(a.out/'JR08_Tile_Edge_Context.csv',borders);write_json(a.out/'JR08_Saved_Event_Adjudication.json',associations)
    render_gallery(z,mainrows,authors,a.out)
    for f in [a.core_zip,a.jr06_zip,a.author_record]:assert sha(f.read_bytes())==before[str(f)]
    write_json(a.out/'JR08_Review_And_Render_Check.json',{'status':'PASS_FOR_RECORDED_AUTHOR_COMMENTS_SOURCE_ADJUDICATION_AND_RENDER_ONLY','panel_comments':12,'complete_five_tag_rubric':False,'actual_inspection_date':author['actual_inspection_date'],'archival_ai_judgments_changed':False,'scientific_masks_scores_assignments_changed':False,'core_payloads_checked':len(copied),'main_order_preserved':True,'tile_boundary_context_checks':2,'reference_only_panels':['d','l'],'repeated_reference_panels':['c','g'],'no_model_or_evaluator_execution':True,'inputs_sha256':{p.name:before[str(p)] for p in [a.core_zip,a.jr06_zip,a.author_record]},'closure_note':'Manuscript integration and document QA are separate from this numerical/source check.'})
    print('PASS: 12 author free-text comments; saved identities and RGB preserved. No new scientific evaluation.')
if __name__=='__main__':main()
