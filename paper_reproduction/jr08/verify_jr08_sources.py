#!/usr/bin/env python3
"""Read-only JR08 source/geometry check. No model, evaluator, or original script execution.
Requires Pillow. All author-review fields remain NOT_REVIEWED.
Use original retained ZIPs; --out must not exist. This does NOT close JR08.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, io, json, math, re, string, zipfile
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, __version__ as pillow_version

CORE_HASH = 'cd03ded8f4f7ac0309f1d5fb89f7454569d4f0798414f37d0143594c3feb70a4'
GALLERY_HASH = '74c1408e7f27e6eab943adeab03d7810acebac1831ec30bb4cfee4a613dbed91'
COLORS = {'ground_truth': (0,114,178), 'selected':(204,121,167), 'linked':(230,159,0)}

def sha(b): return hashlib.sha256(b).hexdigest()
def csvrows(b): return list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))
def js(b): return json.loads(b)
def nk(s): return [int(t) if t.isdigit() else t.casefold() for t in re.split(r'(\d+)',s)]
def ids(s): return [x for x in s.split(';') if x]
def primary(rows): return [r for r in rows if r['score_threshold']=='0.5' and r['mask_iou_threshold']=='0.5']
def unique_suffix(z, s):
    n=[x for x in z.namelist() if x.endswith(s)]
    if len(n)!=1: raise ValueError(f'Expected exactly one ZIP member ending {s}: {n}')
    return z.read(n[0])
def write_json(path,obj): path.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
def write_csv(path,rows):
    if not rows: raise ValueError(f'Empty report: {path}')
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--core-zip',type=Path,required=True)
    ap.add_argument('--jr06-zip',type=Path,required=True,help='JR06_Verified_Evidence_Pending_Scope.zip')
    ap.add_argument('--photos-zip',type=Path,required=True,help='Original 5_carts.zip')
    ap.add_argument('--baseline-zip',type=Path,required=True,help='JR07_Closed_Manuscript_Update_EN.zip')
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists(): raise SystemExit('Refusing an existing output path')
    paths=[a.core_zip,a.jr06_zip,a.photos_zip,a.baseline_zip]
    data=[p.read_bytes() for p in paths]
    before=[sha(b) for b in data]
    checks=[]
    def ck(condition, label):
        if not condition: raise AssertionError(label)
        checks.append(label)
    ck(before[0]==CORE_HASH,'Exact incoming core ZIP identity')
    z,z6,zp,zb=[zipfile.ZipFile(io.BytesIO(b)) for b in data]
    for name,zz in [('core',z),('prior JR06',z6),('photos',zp),('JR07',zb)]:
        ck(zz.testzip() is None,f'{name} ZIP CRC')
        ck(len(zz.namelist())==len(set(zz.namelist())),f'{name} unique member names')
    manifest=js(z.read('COLLECTION_MANIFEST.json'))
    copied=[r for r in manifest['files'] if r['status']=='COPIED']
    ck(len(copied)==143,'143 copied payload records')
    for r in copied:
        b=z.read(r['archive_member'])
        ck(len(b)==r['bytes'] and sha(b)==r['sha256'],'Collector binding: '+r['archive_member'])
    omitted=[r for r in manifest['files'] if r['status']=='NOT_COPIED_SIZE_LIMIT']
    ck(len(omitted)==1 and Path(omitted[0]['original_locator']).name=='failure_case_contact_sheet_highres.pdf', 'Only omitted large file is 30-image high-res contact PDF')
    originals=csvrows(z.read('render/source_package_member_sha256.csv'))
    ck(len(originals)==78,'Original gallery manifest has 78 source members')
    for r in originals:
        b=z.read('evidence/'+r['source_relative_path'])
        ck(sha(b)==r['zip_member_sha256']==r['source_sha256_before']==r['source_sha256_after'] and len(b)==int(r['size_bytes']), 'Sealed source: '+r['source_relative_path'])
    renderrows=csvrows(z.read('render/render_input_sha256.csv'))
    ck(len(renderrows)==68,'68 recorded render inputs')
    for r in renderrows:
        b=z.read('evidence/'+r['source_relative_path'])
        ck(sha(b)==r['sha256_before']==r['sha256_after'],'Render input: '+r['source_relative_path'])
        if r['cropped_rgb_sha256']:
            im=Image.open(io.BytesIO(b)).convert('RGB')
            box=tuple(int(r[k]) for k in ['crop_x0','crop_y0','crop_x1','crop_y1'])
            crop=im.crop(box)
            ck(sha(crop.tobytes())==r['cropped_rgb_sha256'],'Display-only header crop: '+r['candidate_id'])
    sources=csvrows(z.read('evidence/source_manifest_sha256.csv'))
    sbyrole={r['source_role']:r for r in sources}
    sbyhash={r['sha256_before']:r for r in sources}
    for r in sources: ck(r['sha256_before']==r['sha256_after'],'Recorded source unchanged: '+r['source_artifact_id'])
    bound=[]
    def bind(b,role):
        r=sbyrole[role]; ck(sha(b)==r['sha256_before'], 'Prior original binding: '+role)
        bound.append({'role':role,'source_sha256':sha(b),'bytes':len(b),'source_locator':r['absolute_path']})
        return b
    audit='JR06_Verified_Evidence_Pending_Scope/inputs/core/mask_study/results/run_20260803_184228_8PwhfM/'
    files={}
    for name,role in [('prediction_errors.csv','polygon_prediction_errors'),('ground_truth_errors.csv','polygon_ground_truth_errors'),('topology_events.csv','polygon_topology_events'),('matched_pairs.csv','polygon_matched_pairs')]:
        files[name]=csvrows(bind(z6.read(audit+name),role))
    coco=js(bind(z6.read(audit+'frozen_inputs/instances_default.json'),'polygon_ground_truth_coco'))
    predrows={}; tiles={};pred_sources={}
    for card in ['IMG_9459','IMG_9486','IMG_9487']:
        prefix='JR06_Verified_Evidence_Pending_Scope/inputs/core/effort/'+card[4:]+'/'
        b=bind(z6.read(prefix+'detections.csv'),'saved_predictions:'+card);pred_sources[card]=sha(b)
        ix=csvrows(bind(z6.read(prefix+'tiles_index.csv'),'saved_tile_index:'+card))
        for r in ix: tiles[Path(r['tile_name']).name]=(int(r['x']),int(r['y']))
        for r in csvrows(b):
            key=Path(r['image']).name+':'+r['id'];ck(key not in predrows,'Unique prediction key: '+key);predrows[key]=r
    image_cards={int(r['id']):Path(r['file_name']).stem for r in coco['images']}
    gt={f"gt:{r['id']}":r for r in coco['annotations']}
    ck(len(gt)==416,'416 unique reference polygons')
    pe=primary(files['prediction_errors.csv']);ge=primary(files['ground_truth_errors.csv']);te=primary(files['topology_events.csv']);mp=primary(files['matched_pairs.csv'])
    pm={r['prediction_key']:r for r in pe};gm={r['ground_truth_key']:r for r in ge};mm={r['ground_truth_key']:r for r in mp}
    ck((len(pe),len(ge),len(mp))==(55,73,343),'Saved primary unmatched/matched populations')
    elig=csvrows(z.read('evidence/failure_case_eligibility_manifest.csv'))
    ck(Counter(r['eligibility_class'] for r in elig)==dict(A=55,B=73,C=5,D=16,E=3),'152 class records and original denominators')
    for c in 'ABCDE':
        group=[r for r in elig if r['eligibility_class']==c]
        ck(group==sorted(group,key=lambda r:(nk(r['card_id']),nk(r['source_prediction_or_gt_id']))),f'Natural order {c}')
        ck([int(r['eligibility_rank_zero_based']) for r in group]==list(range(len(group))),f'Ranks {c}')
        for i,r in enumerate(group):
            ck(int(r['eligibility_count'])==len(group),'Count for '+c+str(i))
            ck((r['selected_for_main_gallery']=='TRUE')==(i in (0,len(group)-1)),f'Endpoint selection {c}:{i}')
            if c in 'AC':
                sr=pm[r['source_prediction_or_gt_id']]
                ck(r['prediction_id_s']==sr['prediction_key'] and r['saved_assignment_or_event_type']==sr['error_type'],f'{c} outcome {i}')
                for x,y in [('card_id','card'),('saved_prediction_score','prediction_score'),('best_linked_id','best_ground_truth_key'),('saved_best_mask_iou','best_mask_iou')]:
                    if x=='saved_best_mask_iou': x='saved_mask_iou' if 'saved_mask_iou' in r else 'best_saved_mask_iou'
                    ck(r[x]==sr[y],f'{c} {i} {x}')
                if c=='C': ck(sr['error_type']=='DUPLICATE','C is subset of unmatched predictions')
            elif c=='B':
                sr=gm[r['source_prediction_or_gt_id']]
                for x,y in [('card_id','card'),('ground_truth_id_s','ground_truth_key'),('best_linked_id','best_prediction_key'),('best_saved_mask_iou','best_mask_iou'),('saved_assignment_or_event_type','error_type')]:ck(r[x]==sr[y],f'B {i} {x}')
            else:
                matches=[x for x in te if x['card']==r['card_id'] and x['event_type']==r['saved_assignment_or_event_type'] and x['ground_truth_keys']==r['ground_truth_id_s'] and x['prediction_keys']==r['prediction_id_s']]
                ck(len(matches)==1,f'{c} event {i}')
                ck(r['saved_combined_gt_coverage']==matches[0]['combined_gt_coverage'] and r['saved_cross_tile_flag']==matches[0]['cross_tile'],f'{c} coverage {i}')
    rule=js(z.read('evidence/sampling_rule.json'))
    for fn,key in [('inspection_plan.csv','inspection_plan_sha256'),('failure_case_eligibility_manifest.csv','eligibility_manifest_sha256')]: ck(sha(z.read('evidence/'+fn))==rule[key],'Rule hash: '+fn)
    plan=csvrows(z.read('evidence/inspection_plan.csv')); plans={r['candidate_id']:r for r in plan}
    ck(len(plans)==152,'152 unique inspection queue keys')
    for r in plan:
        match=[x for x in elig if x['eligibility_class']==r['eligibility_class'] and x['eligibility_rank_zero_based']==r['eligibility_rank_zero_based']]
        ck(len(match)==1 and all(r[k]==v for k,v in match[0].items()),'Inspection queue source row: '+r['candidate_id'])
    contact=csvrows(z.read('evidence/failure_case_contact_manifest.csv'));vis=csvrows(z.read('evidence/visual_tag_judgments.csv'));panels=csvrows(z.read('evidence/failure_case_panel_manifest.csv'))
    mainpanels=[r for r in panels if r['document']=='MAIN']
    ck((len(contact),len(vis),len(panels),len(mainpanels))==(30,30,42,12),'30 inspected and 12 displayed panels')
    ordered=sorted(plan,key=lambda r:int(r['inspection_order_zero_based']))
    ck([r['candidate_id'] for r in contact]==[r['candidate_id'] for r in ordered[:30]],'Exact first-30 contact queue')
    first={};vi={};selected=[]
    for r in vis:
        cid=r['candidate_id'];vi[cid]=r
        reader_prefix = r['first_reader_id'].removesuffix('_primary_visual_reader')
        ck(bool(reader_prefix) and r['first_reader_id']==reader_prefix+'_primary_visual_reader' and r['second_reader_id']==reader_prefix+'_independent_visual_reader','Recorded visual reader roles: '+cid)
        conf=[t for t in 'FGHIJ' if r['first_reader_tag_'+t]==r['second_reader_tag_'+t]=='CONFIRMED']
        uncertain=[t for t in 'FGHIJ' if not(r['first_reader_tag_'+t]==r['second_reader_tag_'+t] and r['first_reader_tag_'+t] in ('CONFIRMED','ABSENT'))]
        ck(conf==js(r['consensus_confirmed_tags_json']) and uncertain==js(r['consensus_uncertain_tags_json']),'Recorded consensus arithmetic: '+cid)
        for t in conf: first.setdefault(t,cid)
    ck(first=={'F':'A_0000','H':'B_0072','J':'C_0000','G':'D_0001','I':'B_0018'},'Recorded first-confirmed tags')
    fixed=[r['candidate_id'] for r in ordered[:10]]
    expected=fixed[:]
    for r in vis:
        if r['candidate_id'] in first.values() and r['candidate_id'] not in expected: expected.append(r['candidate_id'])
    ck([r['candidate_id'] for r in mainpanels]==expected,'12-panel fixed-plus-qualitative selection')
    ck([r['panel_label'] for r in mainpanels]==[f'({c})' for c in string.ascii_lowercase[:12]],'Display order a-l')
    photos={}
    pixels=[];forms=[]
    for r in contact:
        cid=r['candidate_id'];q=plans[cid];order=int(r['inspection_order_zero_based']);stem=f'{order:03d}_{cid}'
        b=z.read('evidence/inspection_candidates/'+stem+'.png');md=js(z.read('evidence/inspection_candidates/'+stem+'.json'))
        ck(sha(b)==md['rendered_candidate_sha256'],'PNG metadata hash: '+cid)
        ck(md['candidate_id']==cid and md['inspection_order_zero_based']==order,'Metadata key/order: '+cid)
        card=md['card_id']
        if card not in photos:
            pb=bind(unique_suffix(zp,'/'+card+'.jpg'),'full_card_image:'+card)
            photos[card]=Image.open(io.BytesIO(pb)).convert('RGB')
        ck(photos[card].size==(3024,4032),'Original canvas: '+cid)
        entities=md['entities'];expect=[]
        pids=ids(q['prediction_id_s']);gids=ids(q['ground_truth_id_s']);linked=q['best_linked_id'];c=q['eligibility_class']
        if c=='A': expect=[('prediction',x,'unmatched_prediction') for x in pids]+([('ground_truth',linked,'best_saved_overlap_ground_truth')] if linked else [])
        if c=='B': expect=[('ground_truth',x,'unmatched_ground_truth') for x in gids]+([('prediction',linked,'best_saved_overlap_prediction')] if linked else [])
        if c=='C': expect=[('prediction',x,'extra_duplicate_prediction') for x in pids]+[('ground_truth',linked,'duplicate_affected_ground_truth'),('prediction',mm[linked]['prediction_key'],'matched_representative_prediction')]
        if c=='D': expect=[('ground_truth',x,'split_affected_ground_truth') for x in gids]+[('prediction',x,'split_component_prediction') for x in pids]
        if c=='E': expect=[('ground_truth',x,'merge_affected_ground_truth') for x in gids]+[('prediction',x,'merge_prediction') for x in pids]
        ck([(e['entity_type'],e['entity_id'],e['role']) for e in entities]==expect,'Saved entity/role association: '+cid)
        geometry=[]
        for e in entities:
            if e['entity_type']=='ground_truth':
                g=gt[e['entity_id']];ck(image_cards[g['image_id']]==card,'GT card: '+cid)
                polys=[list(zip(s[::2],s[1::2])) for s in g['segmentation']];color=COLORS['ground_truth'];ck(e['source_sha256']==sbyrole['polygon_ground_truth_coco']['sha256_before'],'GT source: '+cid)
            else:
                pr=predrows[e['entity_id']];ox,oy=tiles[Path(pr['image']).name];s=ast.literal_eval(pr['poly']);polys=[[(float(x)+ox,float(y)+oy) for x,y in zip(s[::2],s[1::2])]]
                ck(e['saved_score']==pr['xgb_p'] and e['source_sha256']==pred_sources[card],'Prediction score/source: '+cid)
                color=COLORS['linked'] if e['role'] in ('best_saved_overlap_prediction','matched_representative_prediction') else COLORS['selected']
            geometry.append((polys,color))
        pts=[p for polys,_ in geometry for poly in polys for p in poly];xmin=min(x for x,y in pts);xmax=max(x for x,y in pts);ymin=min(y for x,y in pts);ymax=max(y for x,y in pts)
        pad=max(96,round(.75*max(xmax-xmin,ymax-ymin)));side=max(384,math.ceil(xmax+pad)-math.floor(xmin-pad),math.ceil(ymax+pad)-math.floor(ymin-pad));left=max(0,min(3024-side,round((xmin+xmax-side)/2)));top=max(0,min(4032-side,round((ymin+ymax-side)/2)));box=(left,top,left+side,top+side)
        ck(list(box)==md['crop_bounds_half_open_xyxy'] and pad==md['requested_display_padding_pixels'],'Source geometry/crop: '+cid)
        raw=photos[card].crop(box);fl=Image.new('RGBA',raw.size,(0,0,0,0));ll=Image.new('RGBA',raw.size,(0,0,0,0));fd=ImageDraw.Draw(fl);ld=ImageDraw.Draw(ll)
        for polys,color in geometry:
            for poly in polys:
                sh=[(round(x-left),round(y-top)) for x,y in poly];fd.polygon(sh,fill=(*color,52));ld.line(sh+[sh[0]],fill=(*color,255),width=max(3,raw.width//128),joint='curve')
        overlay=Image.alpha_composite(Image.alpha_composite(raw.convert('RGBA'),fl),ll).convert('RGB')
        png=Image.open(io.BytesIO(b)).convert('RGB');ck(png.size==(1460,810),'Source panel canvas: '+cid)
        rgood=png.crop((10,80,730,800)).tobytes()==raw.resize((720,720),Image.Resampling.LANCZOS).tobytes()
        ogood=png.crop((730,80,1450,800)).tobytes()==overlay.resize((720,720),Image.Resampling.LANCZOS).tobytes()
        ck(rgood and ogood,'Exact photographed and saved-overlay display pixels: '+cid)
        pixels.append({'candidate_id':cid,'inspection_order':order,'raw_crop_pixels_equal':rgood,'overlay_pixels_equal':ogood,'source_png_sha256':sha(b),'original_photo_sha256':md['raw_image_sha256'],'crop_xyxy':json.dumps(list(box))})
        for pr in [x for x in panels if x['candidate_id']==cid]:
            ck([int(pr[k]) for k in ['crop_x0','crop_y0','crop_x1','crop_y1']]==list(box),'Panel manifest crop: '+pr['panel_id'])
            ck(js(pr['visual_tags_confirmed_json'])==js(vi[cid]['consensus_confirmed_tags_json']) and js(pr['visual_tags_uncertain_json'])==js(vi[cid]['consensus_uncertain_tags_json']),'Panel archived tags: '+pr['panel_id'])
            ck(js(pr['overlay_prediction_ids_json'])==[e['entity_id'] for e in entities if e['entity_type']=='prediction'] and js(pr['overlay_gt_ids_json'])==[e['entity_id'] for e in entities if e['entity_type']=='ground_truth'],'Panel overlay entities: '+pr['panel_id'])
    nz=[n for n in zb.namelist() if Path(n).name.startswith('03_Source_') and n.endswith('.zip')];ck(len(nz)==1,'One current JR07 source archive')
    zs=zipfile.ZipFile(io.BytesIO(zb.read(nz[0])));bg=unique_suffix(zs,'/fig_failure_case_gallery_manuscript.pdf');cg=z.read('render/fig_failure_case_gallery_manuscript.pdf')
    ck(bg==cg and sha(cg)==GALLERY_HASH,'Current JR07 gallery exact identity')
    mainmap=[]
    for r in mainpanels:
        q=plans[r['candidate_id']]
        mainmap.append({'panel':r['panel_label'],'panel_id':r['panel_id'],'candidate_id':r['candidate_id'],'card':r['card_id'],'saved_outcome':r['outcome_event_label'],'source_record_id':r['source_prediction_or_gt_id'],'overlay_gt_ids':r['overlay_gt_ids_json'],'overlay_prediction_ids':r['overlay_prediction_ids_json'],'best_saved_mask_iou':q['best_saved_mask_iou'],'archived_present_tags':r['visual_tags_confirmed_json'],'archived_uncertain_tags':r['visual_tags_uncertain_json'],'inspection_png':f"{int(r['inspection_order_zero_based']):03d}_{r['candidate_id']}.png",'human_review_status':'NOT_REVIEWED'})
    ck(mainmap[2]['source_record_id']==mainmap[6]['source_record_id']=='gt:6','Panels c/g share the same reference insect gt:6')
    for path,h in zip(paths,before):ck(sha(path.read_bytes())==h,'Original input ZIP unchanged: '+path.name)
    a.out.mkdir(parents=True)
    write_csv(a.out/'JR08_Panel_Source_Map_EN.csv',mainmap)
    write_csv(a.out/'JR08_Display_Pixel_Comparison.csv',pixels)
    write_csv(a.out/'JR08_Original_Source_Binding.csv',bound)
    write_csv(a.out/'JR08_Collection_Omissions.csv',[{k:r.get(k,'') for k in ('archive_member','original_locator','status','bytes')} for r in manifest['files'] if r['status']!='COPIED'])
    result={'task':'JR08','task_status':'OPEN_AWAITING_AUTHOR_PANEL_REVIEW_AND_INTEGRATION','technical_source_checks':'PASS','assertions_passed':len(checks),'payloads_verified':143,'original_gallery_members_verified':78,'render_inputs_verified':68,'eligibility_records_bound':152,'source_panel_png_json_pairs':30,'main_panels':12,'raw_overlay_pixel_pairs_equal':30,'human_review':'NOT_SUPPLIED_NOT_CERTIFIED','scientific_re_evaluation':'NOT_RUN','original_project_code_execution':'NOT_RUN','generative_image_editing':'NOT_USED','pillow_version':pillow_version,'missing_large_contact_pdf':omitted[0],'input_zip_sha256':dict(zip([p.name for p in paths],before))}
    write_json(a.out/'JR08_TECHNICAL_CHECK.json',result)
    write_json(a.out/'CHECKS.json',checks)
    print(json.dumps({k:result[k] for k in ['task_status','technical_source_checks','assertions_passed','raw_overlay_pixel_pairs_equal']},indent=2))
if __name__=='__main__': main()
