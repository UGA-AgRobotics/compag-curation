#!/usr/bin/env python3
"""JR07: independently check RETAINED runtime records; never run a profiler/model.

Standard library only. Input bytes are pinned. Output must be a fresh directory.
A passing result is arithmetic/source-record consistency, not a new timing run,
hardware authenticity certification, prediction validation, or journal acceptance.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, json, math, statistics, sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

CHECKS: list[dict] = []
STAGES = [1, 2, 3, 4, 5, 6, 7, 10]

def require(ok: bool, name: str, detail=None) -> None:
    CHECKS.append({'check': name, 'passed': bool(ok), 'detail': detail})
    if not ok:
        raise ValueError(f'FAILED: {name}: {detail}')

def same(a, b, name: str, tol=1e-10) -> None:
    if a is None or b is None:
        require(a is b, name, [a, b]); return
    x, y = float(a), float(b)
    require(math.isfinite(x) and math.isfinite(y) and abs(x-y) <= tol,
            name, {'observed': x, 'recorded': y, 'absolute_difference': abs(x-y)})

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader=csv.DictReader(stream); return list(reader), reader.fieldnames

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_json(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def write_csv(path, rows, fields=None):
    fields=fields or list(rows[0])
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)

def aggregate(values):
    return {'first':values[0],'warm_median':statistics.median(values[1:]),
            'warm_min':min(values[1:]),'warm_max':max(values[1:])}

def run(inputs: Path, out: Path):
    inputs=inputs.resolve();out=out.resolve()
    if out.exists(): raise ValueError('Output already exists; use a new directory')
    if inputs==out or inputs in out.parents:raise ValueError('Output must be outside inputs')
    out.mkdir(parents=True)
    pinned=read_json(inputs/'INPUTS_PINNED.json')
    for rec in pinned['files']:
        p=inputs/rec['path']
        require(p.is_file(), f"input_exists:{rec['path']}")
        require(p.stat().st_size==rec['bytes'] and digest(p)==rec['sha256'],f"input_identity:{rec['path']}")
    r=inputs/'core/runtime'
    col=read_json(inputs/'core/COLLECTION_MANIFEST.json')
    copied=[a for a in col['files'] if a.get('status')=='COPIED']
    require(len(copied)==63==col['payload_count'],'collector_payload_count')
    for rec in copied:
        p=inputs/'core'/rec['archive_member']
        require(p.stat().st_size==rec['bytes'] and digest(p)==rec['sha256'],f"collector:{rec['archive_member']}")
    original,_=read_csv(r/'evidence_file_manifest.csv')
    available=[];absent=[]
    for rec in original:
        p=r/rec['relative_path']
        if p.is_file():
            require(p.stat().st_size==int(rec['size_bytes']) and digest(p)==rec['sha256'],f"original_manifest:{rec['relative_path']}")
            available.append(rec['relative_path'])
        else:absent.append(rec)
    write_csv(out/'ORIGINAL_MANIFEST_REFERENCES_NOT_BUNDLED.csv',absent,list(original[0]))
    raw,_=read_csv(r/'runtime_profile_raw.csv')
    summary=read_json(r/'runtime_profile_summary.json')
    execution=read_json(r/'raw_command_outputs/profile_execution.json')
    boundaries=read_json(r/'stage_definition_and_boundary.json')
    require(len(raw)==32 and len({(x['pass_label'],x['stage_id']) for x in raw})==32,'32_unique_stage_pass_rows')
    require(len({x['process_id'] for x in raw})==1,'single_recorded_process')
    require({int(x['process_id']) for x in raw}=={execution['process_id']},'process_identity_receipt')
    require(execution['status']=='PASS' and (r/'raw_command_outputs/profile_exit_code.txt').read_text().strip()=='0','recorded_success_and_exit')
    require(execution['required_stage_rows']==32 and execution['required_stage_ids']==STAGES,'recorded_stage_population')
    require(execution['threshold_applied'] is False and execution['scientific_training_calls']==0 and execution['metric_calculations']==0,'recorded_scope_no_training_threshold_metrics')
    init=summary['initialization']
    require(execution['initialization']==init,'initialization_receipts_agree')
    same(Decimal(init['end_monotonic_ns']-init['start_monotonic_ns'])/Decimal(10**9),init['elapsed_seconds'],'initialization_duration',1e-12)
    before=execution['input_and_checkpoint_hashes_before']; after=execution['input_and_checkpoint_hashes_after']
    require(before==after,'recorded_component_identities_unchanged')
    code_bound=0
    for rel,h in before.items():
        if (r/rel).is_file():require(digest(r/rel)==h,f'execution_component:{rel}');code_bound+=1
    stage_table=[]; pass_table=[]
    for i in range(1,5):
        rr=[x for x in raw if int(x['pass_index'])==i]
        require([int(x['stage_id']) for x in rr]==STAGES,f'stage_order:{i}')
        require({x['pass_class'] for x in rr}==({'FIRST_PROCESS_PASS'} if i==1 else {'WARM_PASS'}),f'pass_class:{i}')
        total=rr[-1];a=int(total['start_monotonic_ns']);b=int(total['end_monotonic_ns'])
        require(int(rr[0]['start_monotonic_ns'])==a and a>init['end_monotonic_ns'],f'direct_start_after_initialization:{i}')
        subtotal=Decimal(0)
        for j,row in enumerate(rr):
            sid=int(row['stage_id']);sa=int(row['start_monotonic_ns']);sb=int(row['end_monotonic_ns'])
            require(a<=sa<sb<=b,f'nested_window:{i}:{sid}')
            require(Decimal(sb-sa)/Decimal(10**9)==Decimal(row['elapsed_seconds']),f'exact_ns_duration:{i}:{sid}')
            require(row['stage_name']==boundaries['definitions'][str(sid)]['name'],f'boundary_name:{i}:{sid}')
            require(row['executed']=='True' and int(row['tile_count'])==48 and int(row['input_width'])==3024 and int(row['input_height'])==4032,f'workload:{i}:{sid}')
            if sid!=1:require(int(row['candidate_count'])==2111,f'candidate_count:{i}:{sid}')
            if sid in (6,7,10):require(int(row['final_feature_dimension'])==74,f'wrapper_width:{i}:{sid}')
            if sid in (7,10):require(int(row['scored_candidate_count'])==2111,f'scored_rows:{i}:{sid}')
            if sid in (2,4):require(row['cuda_synchronization']=='IMMEDIATE_BEFORE_AND_AFTER',f'cuda_sync_record:{i}:{sid}')
            if sid!=10:
                subtotal+=Decimal(row['elapsed_seconds'])
                if j<6:require(sb<=int(rr[j+1]['start_monotonic_ns']),f'nonoverlap_components:{i}:{sid}')
        residual=Decimal(total['elapsed_seconds'])-subtotal
        require(residual>0,f'outer_includes_extra_serialization_overhead:{i}')
        pass_table.append({'pass':total['pass_label'],'direct_seconds':total['elapsed_seconds'],'component_sum_seconds':str(subtotal),'unallocated_outer_seconds':str(residual),'cards_per_hour_arithmetic':3600/float(total['elapsed_seconds'])})
    for sid in STAGES:
        rows=[x for x in raw if int(x['stage_id'])==sid];times=[float(x['elapsed_seconds']) for x in rows]
        stats=aggregate(times);rec=summary['stages'][str(sid)];warm=rec['warm_passes']
        same(stats['first'],rec['first_process_pass']['elapsed_seconds_per_one_card'],f'summary_first:{sid}')
        for key in ['median','min','max']:same(stats[f'warm_{key}'],warm[f'elapsed_seconds_{key}'],f'summary_warm_{key}:{sid}')
        require(warm['n']==3 and len(warm['raw_row_references'])==3,f'3_warm:{sid}')
        for quantity,count in [('tiles',48),('candidates',2111 if sid!=1 else None)]:
            rates=[count/t for t in times] if count else None
            same(rates[0] if rates else None,rec['first_process_pass'][f'{quantity}_per_second'],f'first_rate:{sid}:{quantity}')
            same(statistics.median(rates[1:]) if rates else None,warm[f'{quantity}_per_second_median'],f'warm_rate:{sid}:{quantity}')
            if rates:
                same(min(rates[1:]),warm[f'{quantity}_per_second_observed_range']['min'],f'rate_min:{sid}:{quantity}')
                same(max(rates[1:]),warm[f'{quantity}_per_second_observed_range']['max'],f'rate_max:{sid}:{quantity}')
        stage_table.append({'stage_id':sid,'stage_name':rows[0]['stage_name'],**stats,'device':rows[0]['device'],'unit':'seconds per this card'})
    require(sum(int(x['scored_candidate_count']) for x in raw if x['stage_id']=='7')==execution['timing_only_r92_inference_calls']==8444,'8444_recorded_calls')
    # Reconstruct the original phase-labelled sample summaries; not true hardware maxima.
    samples,_=read_csv(r/'resource_quiescence_and_sampling.csv')
    require(len(samples)==execution['resource_sample_count']==34943,'sample_count')
    resource_rows=[]; window_audit=[]; unavailable_sample_fields=[]
    for row in raw:
        p=row['pass_label']; sid=row['stage_id']
        ss=[x for x in samples if x['pass_label']==p and (sid=='10' or x['stage_id']==sid)]
        a=int(row['start_monotonic_ns']);b=int(row['end_monotonic_ns'])
        outside=[x for x in ss if x['monotonic_ns'] and not a<=int(x['monotonic_ns'])<=b]
        window_audit.append({'pass':p,'stage':sid,'labelled_samples':len(ss),'sample_starts_outside_timer':len(outside),'scope':'retained phase labels; sampler acquisition is not atomic'})
        for src,dst in [('process_rss_bytes','rss_sampled_peak_bytes'),('device_memory_used_mib','device_wide_peak_memory_used_mib'),('device_utilization_percent','device_wide_peak_utilization_percent')]:
            values=[float(x[src]) for x in ss if x[src]!=''];observed=max(values) if values else None
            expected=float(row[dst]) if row[dst]!='' else None
            if expected is None:
                # Original sampler appends a row only after all acquisition calls.
                # A completed CSV can contain a late row absent from the earlier
                # finish_stage snapshot. Preserve the original unavailable field;
                # do not promote a late observation to an accepted original peak.
                unavailable_sample_fields.append({'pass':p,'stage':sid,'field':dst,
                    'original_peak':None,'completed_log_max':observed,
                    'disposition':'ORIGINAL_NOT_AVAILABLE_RETAINED; sample completion time not logged'})
            else:
                same(observed,expected,f'sampled_peak:{p}:{sid}:{src}')
        if sid!='10':
            for kind in ['allocated','reserved']:
                delta=max(0,int(row[f'torch_peak_{kind}_bytes'])-int(row[f'torch_model_baseline_{kind}_bytes']))
                same(delta,row[f'torch_peak_delta_{kind}_bytes'],f'memory_delta:{p}:{sid}:{kind}',0)
        else:
            child=[x for x in raw if x['pass_label']==p and x['stage_id']!='10']
            for key in ['torch_peak_allocated_bytes','torch_peak_reserved_bytes','torch_peak_delta_allocated_bytes','torch_peak_delta_reserved_bytes']:
                same(max(int(x[key]) for x in child),row[key],f'outer_memory_derived_max:{p}:{key}',0)
    stamped=[int(x['monotonic_ns']) for x in samples if x['monotonic_ns']]
    # Multiple acquisition threads are not assumed; record observed sorted intervals.
    require(all(b>a for a,b in zip(stamped,stamped[1:])),'monotonic_sample_starts')
    intervals=[(b-a)/1e9 for a,b in zip(stamped,stamped[1:])]
    stats_sampling={'samples_total':len(samples),'error_rows':sum(bool(x['sampling_error']) for x in samples),'interval_s_min':min(intervals),'interval_s_median':statistics.median(intervals),'interval_s_max':max(intervals),'sampling_is_continuous':False,'nvidia_smi_wait_after_acquisition_s':0.2}
    for sid,field,divisor,title in [(2,'torch_peak_allocated_bytes',2**20,'Proposal stage: Torch allocated MiB'),(4,'torch_peak_allocated_bytes',2**20,'Embedding stage: Torch allocated MiB'),(10,'device_wide_peak_memory_used_mib',1,'Pass: sampled device-wide MiB'),(10,'rss_sampled_peak_bytes',2**30,'Pass: sampled process RSS GiB'),(2,'torch_peak_reserved_bytes',2**20,'Uninterpreted reserved counter MiB')]:
        vals=[float(x[field])/divisor for x in raw if int(x['stage_id'])==sid]
        resource_rows.append({'measurement':title,**aggregate(vals)})
    device_line=next(csv.reader((r/'raw_command_outputs/gpu_device_before_profile.csv').open()))
    require('RTX 4090' in device_line[2],'recorded_gpu')
    capacity=float(device_line[7]); require(capacity==24564,'recorded_gpu_capacity_MiB')
    reserved=resource_rows[-1]['first'];require(reserved>capacity,'reserved_counter_capacity_anomaly_retained')
    # Workload inspection only; no labels, thresholds, metric calculations or model calls.
    work=[]; first_scored=None; first_unscored=None
    model=read_json(inputs/'prior_r92/MODEL_SCHEMA_IMPUTER.json');expected_features=model['features']
    require(len(expected_features)==93,'prior_verified_r92_width')
    require(model['source']['sha256']==before['frozen_assets/cj_ultra_tilesafe_xgb_r92_hybrid.pkl'],'r92_prior_identity_record')
    for i in range(1,5):
        rr=[x for x in raw if int(x['pass_index'])==i];p=rr[0]['pass_label'];d=r/'profile_outputs'/p
        un,uncols=read_csv(d/'candidate_table_unscored.csv');sc,sccols=read_csv(d/'candidate_table_scored.csv')
        require(len(un)==len(sc)==2111 and len(uncols)==77 and sccols==uncols+['r92_probability'],f'candidate_tables_population:{p}')
        require(len({(x['tile_name'],x['candidate_id']) for x in un})==2111,f'unique_candidate_keys:{p}')
        require(all({k:x[k] for k in uncols}==y for x,y in zip(sc,un)),f'scored_retains_predictor_bytes:{p}')
        require(all(math.isfinite(float(x['r92_probability'])) for x in sc),f'finite_saved_probabilities:{p}')
        source_counts=Counter(x['proposal_source'] for x in un)
        require(set(source_counts)<= {'auto','prompt'},f'known_proposal_sources:{p}')
        rowsizes={6:(d/'candidate_table_unscored.csv').stat().st_size,7:(d/'candidate_table_scored.csv').stat().st_size,10:(d/'candidate_table_scored.csv').stat().st_size}
        for sid,size in rowsizes.items():require(int(next(x for x in rr if int(x['stage_id'])==sid)['output_bytes'])==size,f'output_bytes:{p}:{sid}')
        hsc=digest(d/'candidate_table_scored.csv');hun=digest(d/'candidate_table_unscored.csv')
        if i==1:first_scored=hsc;first_unscored=hun
        require(hsc==first_scored and hun==first_unscored,f'crosspass_saved_tables_identical:{p}')
        work.append({'pass':p,'candidates':len(un),'tiles_with_candidates':len({x['tile_name'] for x in un}),'auto':source_counts.get('auto',0),'prompt':source_counts.get('prompt',0),'wrapper_predictors':74,'scored_sha256':hsc,'unscored_sha256':hun})
    wrapper=uncols[3:]
    missing=[x for x in expected_features if x not in wrapper];extra=[x for x in wrapper if x not in expected_features]
    feature_scope={'wrapper_field_count':len(wrapper),'archived_model_schema_count':93,'shared_field_count':len(set(wrapper)&set(expected_features)),'absent_model_fields_zero_filled_by_adapter':missing,'wrapper_fields_not_requested_by_model':extra,'provenance':'93-field schema reuses JR02 inspection, not a new model load','validation_of_prediction_quality':False}
    # Separate Full timing context already supplied for JR05; never pool with AM01.
    a=inputs/'prior_ablation/ablation_run'; full=read_json(a/'timing/full_new_reference.json')
    res,_=read_csv(a/'tables/training_resource_timing.csv');fullrow=next(x for x in res if x['variant_id']=='full_new_reference')
    for k,v in full.items():
        if isinstance(v,(int,float)):same(v,fullrow[k],f'Full_envelope_table:{k}')
        elif k in fullrow:require(str(v)==str(fullrow[k]),f'Full_envelope_identity:{k}')
    # Side-by-side correspondence with earlier T012 is additional, not source substitution.
    prior=read_json(inputs/'core/tasks/T012/audit/EVIDENCE.json')
    same(prior['sampling_interval_seconds']['median'],stats_sampling['interval_s_median'],'prior_T012_sample_interval')
    same(prior['controlled_full_separate']['wall_seconds'],full['wall_seconds'],'prior_T012_Full_timer')
    write_csv(out/'JR07_Runtime_Stages_Verified.csv',stage_table)
    write_csv(out/'JR07_Runtime_Passes_Verified.csv',pass_table)
    write_csv(out/'JR07_Resources_Verified.csv',resource_rows)
    write_csv(out/'JR07_Workload_Verified.csv',work)
    write_csv(out/'JR07_Sample_Window_Audit.csv',window_audit)
    write_json(out/'JR07_Sampling_Verified.json',{**stats_sampling,'original_unavailable_fields':unavailable_sample_fields})
    write_json(out/'JR07_Profile_Feature_Adapter.json',feature_scope)
    write_json(out/'JR07_Full_Envelope_Separate.json',{'record':full,'scope':'fit plus post-fit integrity/validation/serialization; preparation and resampling precede timer','historical_r92_fit_duration':False,'gpu_monitor_process_vs_global_scope':'not persisted','new_training_executed':False})
    # Re-hash after computation; no mutable original data.
    for rec in pinned['files']:require(digest(inputs/rec['path'])==rec['sha256'],f"input_unchanged:{rec['path']}")
    result={'status':'SCOPED_SAVED_RUNTIME_CHECKS_PASS','input_files_pinned':len(pinned['files']),'incoming_payloads_checked':len(copied),'original_runtime_manifest_available_checked':len(available),'original_runtime_manifest_unbundled_records':len(absent),'execution_component_files_present_and_hash_checked':code_bound,'raw_rows':len(raw),'resource_sample_rows':len(samples),'unavailable_original_sample_peak_fields':len(unavailable_sample_fields),'checks_passed':len(CHECKS),'new_timing_experiments':0,'model_loads_or_predictions':0,'historical_device_state_recreated':False,'manuscript_integration':'separate controlled step','scope':'numeric reconstruction from retained records; raw Torch peak counters not independently remeasured','reserved_counter_warning':f'{reserved:g} MiB exceeds recorded device total {capacity:g} MiB; physical interpretation unresolved; not used as VRAM requirement'}
    write_json(out/'VERIFICATION_RESULT.json',result);write_json(out/'CHECKS.json',CHECKS)
    print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    existed_before=args.out.exists()
    try:run(args.inputs,args.out)
    except Exception as exc:
        if not existed_before and args.out.is_dir():write_json(args.out/'FAILED_CHECK.json',{'status':'FAIL','error':str(exc),'checks':CHECKS})
        print(str(exc),file=sys.stderr);return 1
    return 0
if __name__=='__main__':sys.exit(main())
