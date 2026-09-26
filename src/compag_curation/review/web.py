"""Dependency-free, loopback-only web client for human review."""

from __future__ import annotations

import hmac
import json
import secrets
import shutil
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol, Sequence
from urllib.parse import parse_qs, urlsplit

from compag_curation.public_io import PublicIOError, strict_json_bytes
from compag_curation.review.session import MAX_JSON_BODY_BYTES, ReviewSession


_HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="referrer" content="no-referrer">
  <title>COMPAG Full-Image Review</title>
  <style nonce="__NONCE__">
    :root{color-scheme:dark;--bg:#07111f;--panel:#0e1a2b;--line:#29405f;--text:#eef6ff;--muted:#9db0c8;--cyan:#42d3c8;--green:#4dde91;--red:#ff7087;--amber:#ffcc66;--purple:#ad8cff;--shadow:0 18px 50px #02071299}
    *{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 16% -10%,#173154 0,transparent 38%),var(--bg);color:var(--text);font:14px/1.4 Inter,system-ui,sans-serif;min-height:100vh}
    button,input,select{font:inherit}button{cursor:pointer}.app{min-height:100vh;display:grid;grid-template-rows:auto 1fr}
    header{display:grid;grid-template-columns:minmax(250px,1fr) minmax(280px,620px) auto;gap:24px;align-items:center;padding:18px 24px;border-bottom:1px solid var(--line);background:#091525e8;backdrop-filter:blur(16px);position:sticky;top:0;z-index:5}
    .brand{display:flex;align-items:center;gap:13px}.mark{width:40px;height:40px;border-radius:12px;background:linear-gradient(145deg,var(--cyan),#388cff);box-shadow:0 0 28px #42d3c855;display:grid;place-items:center;color:#06121f;font-weight:900}.brand h1{font-size:17px;margin:0}.brand small{display:block;color:var(--muted);margin-top:2px}
    .progressWrap{min-width:0}.progressText{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-bottom:7px}.progress{height:9px;background:#07101d;border:1px solid #213754;border-radius:20px;overflow:hidden}.progress>i{display:block;width:0;height:100%;background:linear-gradient(90deg,#2ea8ff,var(--cyan),var(--green));transition:width .25s}
    .headActions{display:flex;gap:8px}.iconBtn,.softBtn{border:1px solid var(--line);background:#122138;color:var(--text);border-radius:10px;padding:9px 12px}.iconBtn:hover,.softBtn:hover{border-color:#4b6f9e;background:#182b46}.danger{color:#ffb1bd}
    main{min-height:0;display:grid;grid-template-columns:minmax(0,1fr) 370px;gap:16px;padding:16px}.workspace{min-width:0;display:grid;grid-template-rows:auto minmax(540px,1fr);gap:12px}.toolbar,.reviewPanel{background:linear-gradient(180deg,#101e31,#0c1829);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}
    .toolbar{padding:12px;display:flex;flex-wrap:wrap;gap:12px;align-items:center}.toolbar label{color:var(--muted);font-size:12px;display:flex;align-items:center;gap:7px}.toolbar select,.toolbar input[type="text"]{background:#07111f;color:var(--text);border:1px solid var(--line);border-radius:9px;padding:8px 10px;outline:none}.toolbar input:focus,.toolbar select:focus{border-color:var(--cyan)}.layerControls{display:flex;gap:12px;padding-left:10px;border-left:1px solid var(--line)}.search{display:flex;gap:6px;margin-left:auto}.search input{width:190px}
    .viewerShell{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:16px;background-color:#050a12;background-image:linear-gradient(45deg,#0d1725 25%,transparent 25%),linear-gradient(-45deg,#0d1725 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#0d1725 75%),linear-gradient(-45deg,transparent 75%,#0d1725 75%);background-size:20px 20px;background-position:0 0,0 10px,10px -10px,-10px 0;box-shadow:var(--shadow);min-height:540px;touch-action:none}
    .stage{position:absolute;left:0;top:0;transform-origin:0 0;will-change:transform}.stage img,.stage svg{position:absolute;inset:0}.stage img{image-rendering:auto;user-select:none;pointer-events:none}.stage svg{overflow:visible}.proposalMask,.proposalBbox{fill:transparent;vector-effect:non-scaling-stroke;pointer-events:none}.proposalMask{stroke:#62e6df;stroke-width:2}.proposalBbox{stroke:#ffb553;stroke-width:2;stroke-dasharray:8 6}.reviewed{stroke:var(--green)}.modelTarget{stroke:#52dda0}.modelOther{stroke:#ff7890}.humanTarget{stroke:var(--green)}.humanOther{stroke:var(--red)}.humanTargetUncertain{stroke:var(--purple)}.humanOtherUncertain{stroke:#ff9d66}.humanSkip{stroke:#8798ad;stroke-dasharray:5 5}.nonReviewable{opacity:.30}.selected{stroke:var(--amber);stroke-width:4;fill:#ffcc6614;filter:drop-shadow(0 0 5px #ffcc66)}.proposalHit{fill:transparent;stroke:transparent;stroke-width:14;vector-effect:non-scaling-stroke;pointer-events:all;cursor:pointer}.proposalHit:hover~.proposalMask{stroke:white;stroke-width:4}
    .viewerControls{position:absolute;left:12px;top:12px;display:flex;gap:6px;z-index:2}.viewerControls button{width:36px;height:36px;border:1px solid #3c587b;background:#0b182bea;color:var(--text);border-radius:9px;font-size:17px}.viewerHint{position:absolute;left:12px;bottom:12px;color:#c1d3e8;background:#07111fdd;border:1px solid var(--line);padding:7px 10px;border-radius:9px;font-size:12px;pointer-events:none}.sceneBadge{position:absolute;right:12px;bottom:12px;max-width:48%;text-align:right;color:var(--muted);background:#07111fdd;border:1px solid var(--line);padding:7px 10px;border-radius:9px;font-size:11px;pointer-events:none}
    .reviewPanel{padding:16px;display:flex;flex-direction:column;min-height:0}.eyebrow{text-transform:uppercase;letter-spacing:.13em;color:var(--cyan);font-size:10px;font-weight:800}.proposalId{font:700 20px/1.15 ui-monospace,monospace;margin:7px 0 3px;overflow-wrap:anywhere}.sceneName{color:var(--muted);font-size:12px;overflow-wrap:anywhere}.modelNotice{display:none;margin-top:12px;padding:10px;border:1px solid #5e4b84;border-radius:10px;background:#211936;color:#ddcff9;font-size:12px}.modelNotice.show{display:block}.facts{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:16px 0}.fact{padding:10px;border:1px solid var(--line);border-radius:10px;background:#0a1626}.fact b{display:block;font-size:13px;margin-top:3px}.fact span{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
    .currentDecision{min-height:40px;padding:10px 12px;border:1px solid var(--line);border-radius:10px;color:var(--muted);margin-bottom:12px}.actions{display:grid;gap:8px}.choice{border:1px solid transparent;color:#06121f;border-radius:11px;padding:12px 13px;text-align:left;font-weight:800;display:flex;justify-content:space-between;align-items:center}.choice small{font-weight:700;opacity:.72}.target{background:linear-gradient(135deg,#55e8a1,#31bfc1)}.other{background:linear-gradient(135deg,#ff8799,#f5ad72)}.uncertainT{background:linear-gradient(135deg,#d2b2ff,#70b8ff)}.uncertainO{background:linear-gradient(135deg,#ffc875,#ff8ca8)}.skip{background:#233753;color:#d8e6f6;border-color:#3b5679}.choice:hover{filter:brightness(1.08);transform:translateY(-1px)}
    .nav{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}.nav button{padding:10px;border:1px solid var(--line);background:#12233a;color:var(--text);border-radius:10px}.bottom{margin-top:auto;padding-top:16px;display:grid;gap:8px}.toggle{display:flex;align-items:center;gap:8px;color:var(--muted)}.bulkConfirm{border:1px solid #8064a8;background:#241d3d;color:#eadfff;border-radius:11px;padding:11px;font-weight:800}.bulkConfirm:disabled,.finalize:disabled{opacity:.35;cursor:not-allowed}.finalize{border:1px solid #3f8c72;background:#123b34;color:#baffdf;border-radius:11px;padding:11px;font-weight:800}
    .toast{position:fixed;right:20px;bottom:20px;max-width:430px;background:#15263e;border:1px solid #4a6b94;border-radius:12px;padding:12px 15px;box-shadow:var(--shadow);transform:translateY(150%);transition:transform .2s;z-index:20}.toast.show{transform:translateY(0)}.toast.error{border-color:#b84f65}.busy{pointer-events:none;opacity:.65}
    @media(max-width:1040px){header{grid-template-columns:1fr}.headActions{position:absolute;right:18px;top:18px}main{grid-template-columns:1fr}.reviewPanel{min-height:620px}.workspace{grid-template-rows:auto minmax(430px,65vh)}.viewerShell{min-height:430px}.search{margin-left:0}}
  </style>
</head>
<body>
<div class="app" id="app">
  <header>
    <div class="brand"><div class="mark">C</div><div><h1>COMPAG Full-Image Review</h1><small id="reviewContext">Loading verified review state…</small></div></div>
    <div class="progressWrap"><div class="progressText"><span id="progressLabel">Loading…</span><span id="progressPct">0%</span></div><div class="progress"><i id="progressBar"></i></div></div>
    <div class="headActions"><button class="iconBtn" id="undo" title="Undo last decision (0 or Z)" disabled>↶ Undo</button><button class="iconBtn danger" id="close">Close</button></div>
  </header>
  <main>
    <section class="workspace">
      <div class="toolbar">
        <label>Status <select id="filter"><option value="pending">Pending</option><option value="all">All</option><option value="reviewed">Reviewed</option></select></label>
        <label>Group <select id="group"><option value="">All groups</option></select></label>
        <span class="layerControls"><label><input type="checkbox" id="maskOutline" checked> Mask outline</label><label><input type="checkbox" id="bboxToggle"> Rectangle</label><label><input type="checkbox" id="followSelected" checked> Follow selected</label><label id="contextWrap" hidden><input type="checkbox" id="contextToggle"> Model context</label></span>
        <span class="search"><input type="text" id="search" maxlength="64" placeholder="Proposal ID prefix…" aria-label="Proposal ID prefix"><button class="softBtn" id="searchBtn">Find</button></span>
      </div>
      <div class="viewerShell" id="viewer">
        <div class="viewerControls"><button id="zoomIn" title="Zoom in">＋</button><button id="zoomOut" title="Zoom out">−</button><button id="resetView" title="Fit full image">⌂</button></div>
        <div class="stage" id="stage"><img id="sceneImage" alt="Verified full image"><svg id="sceneOverlay" aria-label="Full-image proposal outlines"></svg></div>
        <div class="viewerHint">Wheel to zoom · drag to pan · click a reviewable mask</div><div class="sceneBadge" id="sceneBadge">Full-image scene</div>
      </div>
    </section>
    <aside class="reviewPanel">
      <div class="eyebrow">Selected candidate</div><div class="proposalId" id="proposalId">—</div><div class="sceneName" id="sceneName">—</div><div class="modelNotice" id="modelNotice"></div>
      <div class="facts"><div class="fact"><span>Position</span><b id="position">—</b></div><div class="fact"><span>Group</span><b id="groupName">—</b></div><div class="fact"><span id="metricOneLabel">Predicted IoU</span><b id="iou">—</b></div><div class="fact"><span id="metricTwoLabel">Stability</span><b id="stability">—</b></div></div>
      <div class="currentDecision" id="decision">Pending — no human decision</div>
      <div class="actions">
        <button class="choice target" data-choice="target" disabled><span id="targetChoiceLabel">Target · confident</span><small>1 / A</small></button>
        <button class="choice other" data-choice="other" disabled><span id="otherChoiceLabel">Non-target · confident</span><small>2 / R</small></button>
        <button class="choice uncertainT" data-choice="target_uncertain" disabled><span id="targetUncertainChoiceLabel">Target · uncertain</span><small>4 / W</small></button>
        <button class="choice uncertainO" data-choice="other_uncertain" disabled><span id="otherUncertainChoiceLabel">Non-target · uncertain</span><small>5 / U</small></button>
        <button class="choice skip" data-choice="skip" disabled><span>Skip · zero weight</span><small>3 / S</small></button>
      </div>
      <div class="nav"><button id="previous">← Previous</button><button id="next">Next →</button></div>
      <div class="bottom"><label class="toggle"><input type="checkbox" id="autoNext" checked> Move to next pending candidate after a decision</label><button class="bulkConfirm" id="confirmModelRemainder" hidden disabled>Confirm remaining XGB suggestions · weight 0.4</button><button class="finalize" id="finalize" disabled>Validate and create reviewed CSV</button></div>
    </aside>
  </main>
</div><div class="toast" id="toast" role="status"></div>
<script nonce="__NONCE__">
(()=>{'use strict';
 const params=new URLSearchParams(location.search),token=params.get('token');
 if(!token){document.body.textContent='Missing local review capability token.';return}history.replaceState({},'',location.pathname);
 const $=id=>document.getElementById(id),app=$('app'),toast=$('toast'),svgNS='http://www.w3.org/2000/svg';
 let current=null,summary=null,currentScene=null,imageUrl=null,scale=1,panX=0,panY=0,dragging=false,dragStart=null,busy=false,layerDefaultsApplied=false;
 let presentation={target_label:'Target',non_target_label:'Non-target'};
 let choiceNames={target:'Target · confident',other:'Non-target · confident',target_uncertain:'Target · uncertain',other_uncertain:'Non-target · uncertain',skip:'Skip · zero weight'};
 function message(text,error=false){toast.textContent=text;toast.className='toast show'+(error?' error':'');clearTimeout(message.timer);message.timer=setTimeout(()=>toast.className='toast',4200)}
 function rid(){return crypto.getRandomValues(new Uint32Array(4)).join('-')}
 async function api(path,opts={}){const headers={'X-COMPAG-Review-Token':token,...(opts.headers||{})};if(opts.body)headers['Content-Type']='application/json';const response=await fetch(path,{...opts,headers,cache:'no-store'}),type=response.headers.get('content-type')||'';if(!response.ok){let detail=response.statusText;try{if(type.includes('json'))detail=(await response.json()).error||detail}catch{}throw new Error(detail)}return type.includes('json')?response.json():response.blob()}
 function setBusy(value){busy=value;app.classList.toggle('busy',value)}
 function applyPresentation(value){if(typeof value?.target_label!=='string'||typeof value?.non_target_label!=='string')throw new Error('Review label presentation is invalid.');presentation=value;choiceNames={target:`${value.target_label} · confident`,other:`${value.non_target_label} · confident`,target_uncertain:`${value.target_label} · uncertain`,other_uncertain:`${value.non_target_label} · uncertain`,skip:'Skip · zero weight'};$('targetChoiceLabel').textContent=choiceNames.target;$('otherChoiceLabel').textContent=choiceNames.other;$('targetUncertainChoiceLabel').textContent=choiceNames.target_uncertain;$('otherUncertainChoiceLabel').textContent=choiceNames.other_uncertain}
 function setStageSize(width,height){const stage=$('stage'),image=$('sceneImage'),overlay=$('sceneOverlay');stage.style.width=`${width}px`;stage.style.height=`${height}px`;image.style.width=`${width}px`;image.style.height=`${height}px`;overlay.setAttribute('width',String(width));overlay.setAttribute('height',String(height));overlay.setAttribute('viewBox',`0 0 ${width} ${height}`)}
 function applyTransform(){const viewer=$('viewer').getBoundingClientRect();$('stage').style.transform=`translate(${viewer.width/2+panX}px,${viewer.height/2+panY}px) scale(${scale})`}
 function resetView(){if(!currentScene)return;const viewer=$('viewer'),availableW=Math.max(1,viewer.clientWidth-48),availableH=Math.max(1,viewer.clientHeight-48);scale=Math.max(.02,Math.min(4,availableW/currentScene.width,availableH/currentScene.height));panX=-currentScene.width*scale/2;panY=-currentScene.height*scale/2;applyTransform()}
 function followSelected(scene){if(!scene||!$('followSelected').checked){applyTransform();return}const selected=scene.proposals.find(item=>item.selected===true);if(!selected)throw new Error('Full-image scene has no selected candidate.');const box=geometry(selected.bbox,4,'Selected bounding box'),viewer=$('viewer'),width=Math.max(1,viewer.clientWidth),height=Math.max(1,viewer.clientHeight),padding=Math.min(72,Math.max(24,Math.min(width,height)*.10));let left=width/2+panX+box[0]*scale,top=height/2+panY+box[1]*scale,right=left+box[2]*scale,bottom=top+box[3]*scale;const availableW=Math.max(1,width-2*padding),availableH=Math.max(1,height-2*padding);if(right-left>availableW)panX=-(box[0]+box[2]/2)*scale;else if(left<padding)panX+=padding-left;else if(right>width-padding)panX-=right-(width-padding);if(bottom-top>availableH)panY=-(box[1]+box[3]/2)*scale;else if(top<padding)panY+=padding-top;else if(bottom>height-padding)panY-=bottom-(height-padding);applyTransform()}
 function finiteText(value,digits=4){const number=Number(value);return Number.isFinite(number)?number.toFixed(digits):'Unavailable'}
 function integerText(value){return Number.isSafeInteger(value)?value.toLocaleString():'Unavailable'}
 function exactFlag(value,yes,no){return value===1?yes:(value===0?no:'unavailable')}
 function updateSummary(s){summary=s;const round=s.source_kind==='ACTIVE_LEARNING_ROUND',assisted=s.source_kind==='STAGE20_PUBLISHED_MODEL_ASSIST',reviewed=Number(s.reviewed),remaining=Number(s.remaining),progress=Number(s.progress_percent);if(!Number.isSafeInteger(reviewed)||reviewed<0||!Number.isSafeInteger(remaining)||remaining<0||!Number.isFinite(progress)||assisted&&typeof s.can_confirm_model_remainder!=='boolean')throw new Error('Reviewer status metadata is invalid.');$('reviewContext').textContent=round?`Active Learning round ${integerText(s.round_number)} · verified hard cases · model suggestions are context only`:assisted?'Published XGBoost r92 transfer assist · verify corrections, then explicitly confirm any remainder':'Cold-start full-image labeling · no model prediction · durable autosave';$('progressLabel').textContent=`${reviewed.toLocaleString()} reviewed · ${remaining.toLocaleString()} remaining`;$('progressPct').textContent=`${progress.toFixed(2)}%`;$('progressBar').style.width=`${Math.max(0,Math.min(100,progress))}%`;const bulk=$('confirmModelRemainder');bulk.hidden=!assisted;bulk.disabled=!assisted||s.can_confirm_model_remainder!==true||remaining===0||s.status==='FINALIZED';$('finalize').disabled=remaining!==0||s.status==='FINALIZED';if(s.status==='FINALIZED')enableReviewActions(false);if($('group').options.length===1&&Array.isArray(s.groups))s.groups.forEach(g=>{const o=document.createElement('option');o.value=String(g);o.textContent=String(g);$('group').append(o)})}
 function geometry(values,minimum,role){if(!Array.isArray(values)||values.length<minimum||values.length%2!==0||values.some(v=>!Number.isFinite(Number(v))))throw new Error(`${role} geometry is invalid.`);return values.map(Number)}
 function classesFor(proposal,base){const names=[base];if(proposal.choice)names.push('reviewed');if(!proposal.reviewable)names.push('nonReviewable');if(proposal.model?.prediction===1)names.push('modelTarget');if(proposal.model?.prediction===0)names.push('modelOther');const human={target:'humanTarget',other:'humanOther',target_uncertain:'humanTargetUncertain',other_uncertain:'humanOtherUncertain',skip:'humanSkip'}[proposal.choice];if(human)names.push(human);if(proposal.selected)names.push('selected');return names.join(' ')}
 function svgShape(name,proposal,visibleClass){const poly=geometry(proposal.polygon||[],0,'Mask');let element;if(name==='polygon'&&poly.length>=6){element=document.createElementNS(svgNS,'polygon');element.setAttribute('points',Array.from({length:poly.length/2},(_,i)=>`${poly[i*2]},${poly[i*2+1]}`).join(' '))}else{const box=geometry(proposal.bbox,4,'Bounding box');element=document.createElementNS(svgNS,'rect');element.setAttribute('x',box[0]);element.setAttribute('y',box[1]);element.setAttribute('width',box[2]);element.setAttribute('height',box[3])}element.setAttribute('class',visibleClass);return element}
 function renderScene(scene){const overlay=$('sceneOverlay');while(overlay.firstChild)overlay.firstChild.remove();const round=scene.kind==='VERIFIED_STAGE60_ORIGINAL_FULL_IMAGE',assisted=scene.model?.mode==='PUBLISHED_TRANSFER_ASSIST',hasModel=round||assisted,showContext=$('contextToggle').checked,showMask=$('maskOutline').checked,showBbox=$('bboxToggle').checked;for(const proposal of scene.proposals){if(!proposal.reviewable&&!showContext)continue;const group=document.createElementNS(svgNS,'g');const title=document.createElementNS(svgNS,'title'),model=proposal.model||{};title.textContent=hasModel?`${proposal.label_prefix} · p=${finiteText(model.xgb_p,3)} · uncertainty=${finiteText(model.uncertainty,3)} · model ${exactFlag(model.prediction,presentation.target_label,presentation.non_target_label)}${round?(proposal.reviewable?' · hard case':' · context only'):proposal.choice?' · reviewed':' · pending'}`:`${proposal.label_prefix}${proposal.choice?' · reviewed':' · pending'}`;group.append(title);let hit=null;if(proposal.reviewable){hit=svgShape((proposal.polygon||[]).length>=6?'polygon':'rect',proposal,'proposalHit');hit.setAttribute('aria-label',`Review proposal ${String(proposal.label_prefix)}`);hit.addEventListener('click',event=>{event.stopPropagation();loadProposal(String(proposal.proposal_id))});group.append(hit)}if(showMask&&(proposal.polygon||[]).length>=6)group.append(svgShape('polygon',proposal,classesFor(proposal,'proposalMask')));if(showBbox)group.append(svgShape('rect',proposal,classesFor(proposal,'proposalBbox')));overlay.append(group)}$('contextWrap').hidden=!round;$('sceneBadge').textContent=round?`${integerText(scene.reviewable_proposal_count)} hard cases · ${integerText(scene.proposal_count)} scored candidates`:assisted?`${integerText(scene.proposal_count)} XGB-scored candidates · model-colored outlines`:`${integerText(scene.proposal_count)} candidates · mask-derived outlines`}
 function enableReviewActions(value){document.querySelectorAll('[data-choice]').forEach(button=>button.disabled=!value);$('undo').disabled=!value}
 function clearReview(){current=null;currentScene=null;enableReviewActions(false);$('confirmModelRemainder').disabled=true;if(imageUrl){URL.revokeObjectURL(imageUrl);imageUrl=null}$('sceneImage').removeAttribute('src');const overlay=$('sceneOverlay');while(overlay.firstChild)overlay.firstChild.remove();$('proposalId').textContent='IMAGE UNAVAILABLE';$('proposalId').title='';$('sceneName').textContent='The verified full image could not be loaded.';$('position').textContent='—';$('groupName').textContent='—';$('iou').textContent='—';$('stability').textContent='—';$('decision').textContent='Review actions are disabled until a full image is verified.';$('modelNotice').className='modelNotice';$('finalize').disabled=true}
 async function decodedScene(scene){const blob=await api(`/api/scene?scene=${encodeURIComponent(scene.scene_id)}`),url=URL.createObjectURL(blob),probe=new Image();try{await new Promise((resolve,reject)=>{probe.onload=resolve;probe.onerror=()=>reject(new Error('Verified full-image scene could not be decoded.'));probe.src=url});if(probe.naturalWidth!==scene.width||probe.naturalHeight!==scene.height)throw new Error('Verified full-image dimensions differ from scene metadata.');return url}catch(error){URL.revokeObjectURL(url);throw error}}
 function validateScene(scene,proposalId){if(!scene||typeof scene.scene_id!=='string'||!Number.isSafeInteger(scene.width)||!Number.isSafeInteger(scene.height)||scene.width<1||scene.height<1||!Array.isArray(scene.proposals))throw new Error('Full-image scene metadata is invalid.');const selected=scene.proposals.find(item=>item.proposal_id===proposalId);if(!selected||selected.reviewable!==true)throw new Error('Selected candidate is not reviewable in this scene.');return selected}
 function validateAssistedScene(scene,proposal){if(scene.model?.mode!=='PUBLISHED_TRANSFER_ASSIST'||scene.model?.model_suggestions_are_human_context!==true)throw new Error('Published-model scene binding is invalid.');for(const row of scene.proposals){const model=row.model;if(!model||typeof model.xgb_p!=='number'||!Number.isFinite(model.xgb_p)||model.xgb_p<0||model.xgb_p>1||typeof model.uncertainty!=='number'||!Number.isFinite(model.uncertainty)||model.uncertainty<0||model.uncertainty>.5||model.prediction!==0&&model.prediction!==1)throw new Error('Published-model scene score is invalid.')}const selected=scene.proposals.find(row=>row.proposal_id===proposal.proposal_id),model=selected?.model;if(!model||proposal.prediction!==model.prediction||proposal.xgb_p!==model.xgb_p||proposal.uncertainty!==model.uncertainty)throw new Error('Selected published-model score differs from its scene.');return true}
 async function show(payload){let nextUrl=null;try{const proposal=payload?.proposal,scene=payload?.scene,nextSummary=payload?.summary;if(!proposal||!scene||!nextSummary)throw new Error('Reviewer full-image metadata is incomplete.');validateScene(scene,proposal.proposal_id);if(nextSummary.source_kind==='STAGE20_PUBLISHED_MODEL_ASSIST')validateAssistedScene(scene,proposal);const sceneChanged=!currentScene||currentScene.scene_id!==scene.scene_id;if(sceneChanged)nextUrl=await decodedScene(scene);const round=nextSummary.source_kind==='ACTIVE_LEARNING_ROUND',assisted=nextSummary.source_kind==='STAGE20_PUBLISHED_MODEL_ASSIST',hasModel=round||assisted,decision=proposal.decision,view={id:String(proposal.proposal_id??''),prefix:String(proposal.label_prefix??'Unavailable'),scene:round?`${String(scene.image_name)} · hard-case rank ${Number.isSafeInteger(proposal.selection_rank)?'#'+proposal.selection_rank:'Unavailable'}`:assisted?`${String(scene.image_name)} · model-uncertainty rank ${Number.isSafeInteger(proposal.selection_rank)?'#'+proposal.selection_rank:'Unavailable'}`:`${String(scene.image_name)} · verified full-image mosaic`,position:`${integerText(proposal.position)} / ${integerText(proposal.total)}`,group:String(proposal.group_id??'Unavailable'),metricOne:hasModel?'XGB probability':'Predicted IoU',metricTwo:hasModel?'Uncertainty':'Stability',metricOneValue:finiteText(hasModel?proposal.xgb_p:proposal.predicted_iou),metricTwoValue:finiteText(hasModel?proposal.uncertainty:proposal.stability_score),decision:decision?`Saved: ${choiceNames[decision.choice]??'Unavailable'} · label ${decision.label??'Unavailable'} · weight ${decision.review_weight??'Unavailable'}`:'Pending — no human decision'};const oldUrl=imageUrl;if(nextUrl){imageUrl=nextUrl;$('sceneImage').src=nextUrl;nextUrl=null}currentScene=scene;setStageSize(scene.width,scene.height);if(!layerDefaultsApplied){$('maskOutline').checked=scene.overlay_defaults?.mask_outline===true;$('bboxToggle').checked=scene.overlay_defaults?.bbox===true;layerDefaultsApplied=true}updateSummary(nextSummary);$('proposalId').textContent=view.prefix;$('proposalId').title=view.id;$('sceneName').textContent=view.scene;$('position').textContent=view.position;$('groupName').textContent=view.group;$('metricOneLabel').textContent=view.metricOne;$('metricTwoLabel').textContent=view.metricTwo;$('iou').textContent=view.metricOneValue;$('stability').textContent=view.metricTwoValue;$('decision').textContent=view.decision;const notice=$('modelNotice');if(round){notice.textContent=`Model suggestion: ${exactFlag(proposal.prediction,presentation.target_label,presentation.non_target_label)} at p=${finiteText(proposal.xgb_p,4)}. This is context only; your explicit decision is authoritative.`;notice.className='modelNotice show'}else if(assisted){notice.textContent=`Published r92 suggestion: ${exactFlag(proposal.prediction,presentation.target_label,presentation.non_target_label)} at p=${finiteText(proposal.xgb_p,4)}, uncertainty=${finiteText(proposal.uncertainty,4)}. Correct it with a human decision, or explicitly confirm the remaining suggestions at reduced review weight 0.4.`;notice.className='modelNotice show'}else{notice.textContent='';notice.className='modelNotice'}renderScene(scene);current=proposal;if(sceneChanged)resetView();else followSelected(scene);enableReviewActions(nextSummary.status!=='FINALIZED');if(oldUrl&&oldUrl!==imageUrl)URL.revokeObjectURL(oldUrl)}catch(error){if(nextUrl)URL.revokeObjectURL(nextUrl);clearReview();throw error}}
 async function loadProposal(id){if(busy)return;setBusy(true);try{await show(await api(`/api/proposal?id=${encodeURIComponent(id)}`))}catch(error){message(error.message,true)}finally{setBusy(false)}}
 async function navigate(direction,forceFilter=null){if(busy)return;setBusy(true);try{await show(await api('/api/navigate',{method:'POST',body:JSON.stringify({current:current?.proposal_id||null,direction,state_filter:forceFilter||$('filter').value,group:$('group').value||null})}))}catch(error){message(error.message,true)}finally{setBusy(false)}}
 async function decide(choice){if(busy||!current)return;if(choice==='skip'&&!confirm('Skip this candidate? This explicitly saves label 0, action skip, and review weight 0.0; it will contribute zero training weight.'))return;setBusy(true);try{const saved=await api('/api/decision',{method:'POST',body:JSON.stringify({proposal_id:current.proposal_id,choice,request_id:rid()})});message(`Saved: ${choiceNames[choice]}`);if($('autoNext').checked&&saved.summary.remaining>0){try{await show(await api('/api/navigate',{method:'POST',body:JSON.stringify({current:current.proposal_id,direction:1,state_filter:'pending',group:$('group').value||null})}))}catch(_navigationError){await show(saved);message('Saved. No pending candidate remains in the current group.')}}else await show(saved)}catch(error){message(error.message,true)}finally{setBusy(false)}}
 document.querySelectorAll('[data-choice]').forEach(button=>button.addEventListener('click',()=>decide(button.dataset.choice)));$('previous').onclick=()=>navigate(-1);$('next').onclick=()=>navigate(1);$('filter').onchange=()=>navigate(1);$('group').onchange=()=>navigate(1);for(const id of ['maskOutline','bboxToggle','contextToggle'])$(id).onchange=()=>{if(currentScene)renderScene(currentScene)};
 $('followSelected').onchange=()=>{if(currentScene&&$('followSelected').checked)followSelected(currentScene)};
 $('undo').onclick=async()=>{if(busy||!current)return;setBusy(true);try{await show(await api('/api/undo',{method:'POST',body:JSON.stringify({request_id:rid()})}));message('Last decision undone')}catch(error){message(error.message,true)}finally{setBusy(false)}};
 $('confirmModelRemainder').onclick=async()=>{if(busy||!current||summary?.source_kind!=='STAGE20_PUBLISHED_MODEL_ASSIST'||summary?.can_confirm_model_remainder!==true)return;const remaining=Number(summary.remaining);if(!Number.isSafeInteger(remaining)||remaining<1){message('No pending XGB suggestion is available to confirm.',true);return}if(!confirm(`Confirm exactly ${remaining.toLocaleString()} remaining XGB suggestions?\n\nEach suggestion will be saved as an uncertain label with review weight 0.4. This is not the same as reviewing each proposal manually.\n\nThis single action is reversible with Undo.`))return;setBusy(true);try{await show(await api('/api/confirm-model-remainder',{method:'POST',body:JSON.stringify({request_id:rid()})}));message(`Confirmed exactly ${remaining.toLocaleString()} model suggestions at review weight 0.4`)}catch(error){message(error.message,true)}finally{setBusy(false)}};
 async function search(){if(busy)return;setBusy(true);try{await show(await api('/api/search',{method:'POST',body:JSON.stringify({prefix:$('search').value})}))}catch(error){message(error.message,true)}finally{setBusy(false)}}$('searchBtn').onclick=search;$('search').onkeydown=event=>{if(event.key==='Enter'){event.preventDefault();search()}};
 $('finalize').onclick=async()=>{if(busy||!confirm('Validate all decisions and create the final reviewed CSV? This publication cannot be overwritten.'))return;setBusy(true);try{const result=await api('/api/finalize',{method:'POST',body:'{}'});message(`Reviewed CSV created: ${result.output}`);updateSummary(await api('/api/status'))}catch(error){message(error.message,true)}finally{setBusy(false)}};
 $('close').onclick=async()=>{if(busy||!confirm('Close the reviewer? All saved decisions will remain available.'))return;setBusy(true);try{await api('/api/close',{method:'POST',body:'{}'});document.body.textContent='Reviewer closed safely. You may close this tab.'}catch(error){message(error.message,true);setBusy(false)}};
 $('zoomIn').onclick=()=>{scale=Math.min(16,scale*1.25);applyTransform()};$('zoomOut').onclick=()=>{scale=Math.max(.01,scale/1.25);applyTransform()};$('resetView').onclick=resetView;
 const viewer=$('viewer');viewer.addEventListener('wheel',event=>{event.preventDefault();const before=scale;scale=Math.max(.01,Math.min(16,scale*(event.deltaY<0?1.12:.89)));const rect=viewer.getBoundingClientRect(),x=event.clientX-rect.left-rect.width/2,y=event.clientY-rect.top-rect.height/2;panX=x-(x-panX)*scale/before;panY=y-(y-panY)*scale/before;applyTransform()},{passive:false});viewer.addEventListener('pointerdown',event=>{if(event.target.closest('.viewerControls')||event.target.classList.contains('proposalHit'))return;dragging=true;dragStart=[event.clientX,event.clientY,panX,panY];viewer.setPointerCapture(event.pointerId)});viewer.addEventListener('pointermove',event=>{if(dragging){panX=dragStart[2]+event.clientX-dragStart[0];panY=dragStart[3]+event.clientY-dragStart[1];applyTransform()}});viewer.addEventListener('pointerup',()=>dragging=false);viewer.addEventListener('pointercancel',()=>dragging=false);
 addEventListener('resize',()=>currentScene?resetView():applyTransform());addEventListener('keydown',event=>{if(busy||event.repeat||event.ctrlKey||event.metaKey||event.altKey||['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName))return;const key=event.key.toLowerCase(),map={'1':'target','a':'target','2':'other','r':'other','4':'target_uncertain','w':'target_uncertain','5':'other_uncertain','u':'other_uncertain','3':'skip','s':'skip'};if(map[key]){event.preventDefault();decide(map[key])}else if(key==='0'||key==='z'){event.preventDefault();$('undo').click()}else if(event.key==='ArrowRight'){event.preventDefault();navigate(1)}else if(event.key==='ArrowLeft'){event.preventDefault();navigate(-1)}});
 (async()=>{try{applyPresentation(await api('/api/presentation'));summary=await api('/api/status');updateSummary(summary);const first=summary.first_pending||await api('/api/first');if(typeof first==='string')await loadProposal(first);else if(first?.proposal_id)await loadProposal(first.proposal_id);else message('No reviewable candidate is available.',true)}catch(error){clearReview();message(error.message,true)}})();
})();
</script></body></html>'''


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


class _ReviewServer(ThreadingHTTPServer):
    MAX_CONCURRENT_HANDLERS = 32
    allow_reuse_address = False
    daemon_threads = True
    block_on_close = True
    request_queue_size = 16

    def __init__(
        self,
        address: tuple[str, int],
        session: ReviewSession,
        token: str,
        nonce: str,
        *,
        target_label: str,
        non_target_label: str,
    ) -> None:
        self.review_session = session
        self.capability_token = token
        self.page_nonce = nonce
        self.presentation = {
            "schema": "compag-curation-review-presentation/v1",
            "target_label": target_label,
            "non_target_label": non_target_label,
        }
        self._handler_slots = threading.BoundedSemaphore(
            self.MAX_CONCURRENT_HANDLERS
        )
        self._handler_count_lock = threading.Lock()
        self._active_handlers = 0
        self._peak_handlers = 0
        super().__init__(address, _ReviewHandler, bind_and_activate=True)

    @property
    def peak_concurrent_handlers(self) -> int:
        with self._handler_count_lock:
            return self._peak_handlers

    @property
    def active_concurrent_handlers(self) -> int:
        with self._handler_count_lock:
            return self._active_handlers

    def _release_handler_slot(self) -> None:
        with self._handler_count_lock:
            self._active_handlers -= 1
        self._handler_slots.release()

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._handler_slots.acquire(blocking=False):
            request.close()
            return
        with self._handler_count_lock:
            self._active_handlers += 1
            self._peak_handlers = max(
                self._peak_handlers,
                self._active_handlers,
            )
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._release_handler_slot()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._release_handler_slot()

    def get_request(self) -> tuple[Any, Any]:
        connection, address = super().get_request()
        connection.settimeout(10.0)
        return connection, address

    def handle_error(self, request: object, client_address: object) -> None:
        error = sys.exc_info()[1]
        if isinstance(error, (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


class _ReviewHandler(BaseHTTPRequestHandler):
    server: _ReviewServer
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _headers(self, status: int, content_type: str, length: int) -> None:
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Connection", "close")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'; "
            f"style-src-elem 'nonce-{self.server.page_nonce}'; style-src-attr 'unsafe-inline'; "
            f"script-src 'nonce-{self.server.page_nonce}'; "
            "connect-src 'self'; img-src 'self' blob: data:",
        )
        self.end_headers()

    def _send(self, status: int, payload: bytes, content_type: str) -> None:
        self._headers(status, content_type, len(payload))
        self.wfile.write(payload)

    def _error(self, status: int, message: str) -> None:
        self._send(status, _json_bytes({"status": "ERROR", "error": message}), "application/json; charset=ascii")

    def _host_ok(self) -> bool:
        expected = f"127.0.0.1:{self.server.server_port}"
        return hmac.compare_digest(self.headers.get("Host", ""), expected)

    def _token_ok(self, query: dict[str, list[str]] | None = None) -> bool:
        supplied = self.headers.get("X-COMPAG-Review-Token", "")
        if not supplied and query is not None:
            values = query.get("token", [])
            supplied = values[0] if len(values) == 1 else ""
        return hmac.compare_digest(supplied, self.server.capability_token)

    def _authorized(self, query: dict[str, list[str]] | None = None, *, require_origin: bool = False) -> bool:
        if not self._host_ok() or not self._token_ok(query):
            self._error(HTTPStatus.FORBIDDEN, "local review authorization failed")
            return False
        if require_origin:
            expected = f"http://127.0.0.1:{self.server.server_port}"
            if not hmac.compare_digest(self.headers.get("Origin", ""), expected):
                self._error(HTTPStatus.FORBIDDEN, "review request origin is invalid")
                return False
        return True

    def _body(self, expected_keys: set[str]) -> dict[str, object] | None:
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "request must use application/json")
            return None
        length_text = self.headers.get("Content-Length", "")
        if not length_text.isdigit() or not (0 <= int(length_text) <= MAX_JSON_BODY_BYTES):
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body size is invalid")
            return None
        payload = self.rfile.read(int(length_text))
        try:
            value = strict_json_bytes(payload, "review API request")
        except PublicIOError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return None
        if not isinstance(value, dict) or set(value) != expected_keys:
            self._error(HTTPStatus.BAD_REQUEST, "review API request fields are invalid")
            return None
        return value

    def _scene_payload(self, value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            raise PublicIOError("review result metadata is invalid")
        proposal = value.get("proposal")
        if not isinstance(proposal, dict):
            raise PublicIOError("review proposal metadata is invalid")
        proposal_id = proposal.get("proposal_id")
        if not isinstance(proposal_id, str):
            raise PublicIOError("review proposal identity is invalid")
        return self.server.review_session.scene_payload(proposal_id)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=False)
        try:
            if parsed.path == "/":
                if not self._authorized(query):
                    return
                payload = _HTML.replace("__NONCE__", self.server.page_nonce).encode("utf-8")
                self._send(HTTPStatus.OK, payload, "text/html; charset=utf-8")
                return
            if not self._authorized():
                return
            if parsed.path == "/api/status" and not query:
                self._send(HTTPStatus.OK, _json_bytes(self.server.review_session.summary()), "application/json; charset=ascii")
            elif parsed.path == "/api/presentation" and not query:
                self._send(HTTPStatus.OK, _json_bytes(self.server.presentation), "application/json; charset=ascii")
            elif parsed.path == "/api/first" and not query:
                summary = self.server.review_session.summary()
                proposal_id = summary["first_pending"] or self.server.review_session.index.order[0]
                self._send(HTTPStatus.OK, _json_bytes({"proposal_id": proposal_id}), "application/json; charset=ascii")
            elif parsed.path == "/api/proposal" and set(query) == {"id"} and len(query["id"]) == 1:
                payload = self.server.review_session.scene_payload(query["id"][0])
                self._send(HTTPStatus.OK, _json_bytes(payload), "application/json; charset=ascii")
            elif parsed.path == "/api/scene" and set(query) == {"scene"} and len(query["scene"]) == 1:
                payload = self.server.review_session.scene_bytes(query["scene"][0])
                self._send(HTTPStatus.OK, payload, "image/png")
            else:
                self._error(HTTPStatus.NOT_FOUND, "review route not found")
        except (PublicIOError, OSError, ValueError) as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        if parsed.query:
            self._error(HTTPStatus.NOT_FOUND, "review route not found")
            return
        if not self._authorized(require_origin=True):
            return
        try:
            if parsed.path == "/api/decision":
                body = self._body({"proposal_id", "choice", "request_id"})
                if body is None:
                    return
                payload = self.server.review_session.set_decision(
                    str(body["proposal_id"]),
                    str(body["choice"]),
                    request_id=str(body["request_id"]),
                )
                payload = self._scene_payload(payload)
            elif parsed.path == "/api/confirm-model-remainder":
                body = self._body({"request_id"})
                if body is None:
                    return
                payload = self.server.review_session.confirm_model_remainder(
                    request_id=str(body["request_id"])
                )
                payload = self._scene_payload(payload)
            elif parsed.path == "/api/undo":
                body = self._body({"request_id"})
                if body is None:
                    return
                payload = self.server.review_session.undo(request_id=str(body["request_id"]))
                payload = self._scene_payload(payload)
            elif parsed.path == "/api/navigate":
                body = self._body({"current", "direction", "state_filter", "group"})
                if body is None:
                    return
                current = body["current"]
                group = body["group"]
                if current is not None and not isinstance(current, str):
                    raise PublicIOError("current proposal is invalid")
                if group is not None and not isinstance(group, str):
                    raise PublicIOError("review group is invalid")
                if type(body["direction"]) is not int:
                    raise PublicIOError("navigation direction is invalid")
                payload = self.server.review_session.navigate(
                    current,
                    body["direction"],
                    state_filter=str(body["state_filter"]),
                    group=group,
                )
                payload = self._scene_payload(payload)
            elif parsed.path == "/api/search":
                body = self._body({"prefix"})
                if body is None:
                    return
                payload = self.server.review_session.search(str(body["prefix"]))
                payload = self._scene_payload(payload)
            elif parsed.path == "/api/finalize":
                body = self._body(set())
                if body is None:
                    return
                payload = self.server.review_session.finalize()
            elif parsed.path == "/api/close":
                body = self._body(set())
                if body is None:
                    return
                payload = {"status": "CLOSING"}
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self._error(HTTPStatus.NOT_FOUND, "review route not found")
                return
            self._send(HTTPStatus.OK, _json_bytes(payload), "application/json; charset=ascii")
        except (PublicIOError, OSError, ValueError) as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))

    def do_PUT(self) -> None:  # noqa: N802
        self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method not allowed")

    do_DELETE = do_PUT
    do_PATCH = do_PUT


def _opener_command(url: str) -> list[str] | None:
    try:
        wsl = "microsoft" in Path("/proc/sys/kernel/osrelease").read_text(encoding="ascii").lower()
    except (OSError, UnicodeError):
        wsl = False
    candidates: list[tuple[str, list[str]]] = []
    if wsl:
        candidates.append(("wslview", ["wslview", url]))
    candidates.extend((("xdg-open", ["xdg-open", url]), ("gio", ["gio", "open", url])))
    for executable, argv in candidates:
        resolved = shutil.which(executable)
        if resolved:
            argv[0] = resolved
            return argv
    return None


class ReviewPageOpener(Protocol):
    def run(self, argv: Sequence[str]) -> None: ...


class LocalReviewPageOpener:
    """Open one local URL without a shell or inherited standard streams."""

    def run(self, argv: Sequence[str]) -> None:
        subprocess.Popen(
            [str(part) for part in argv],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )


def open_local_review_page(
    url: str,
    *,
    opener: ReviewPageOpener | None = None,
) -> bool:
    command = _opener_command(url)
    if command is None:
        return False
    runner = LocalReviewPageOpener() if opener is None else opener
    try:
        runner.run(command)
    except OSError:
        return False
    return True


def _presentation_label(value: str, role: str) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 40
        or value.strip() != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise PublicIOError(f"{role} must be 1-40 visible characters without surrounding whitespace")
    return value


def create_server(
    session: ReviewSession,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    target_label: str = "Target",
    non_target_label: str = "Non-target",
) -> tuple[_ReviewServer, str]:
    if host != "127.0.0.1":
        raise PublicIOError("review web server may bind only to 127.0.0.1")
    if not isinstance(port, int) or not 0 <= port <= 65535:
        raise PublicIOError("review web port must be between 0 and 65535")
    target_label = _presentation_label(target_label, "target label")
    non_target_label = _presentation_label(non_target_label, "non-target label")
    if target_label == non_target_label:
        raise PublicIOError("target and non-target labels must differ")
    token = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(24)
    server = _ReviewServer(
        (host, port),
        session,
        token,
        nonce,
        target_label=target_label,
        non_target_label=non_target_label,
    )
    url = f"http://127.0.0.1:{server.server_port}/?token={token}"
    return server, url


def run_web_review(
    session: ReviewSession,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
    target_label: str = "Target",
    non_target_label: str = "Non-target",
) -> dict[str, Any]:
    server, url = create_server(
        session,
        host=host,
        port=port,
        target_label=target_label,
        non_target_label=non_target_label,
    )
    opened = open_local_review_page(url) if open_browser else False
    public_url = f"http://127.0.0.1:{server.server_port}/"
    print(
        json.dumps(
            {
                "schema": "compag-curation-review-web-start/v1",
                "status": "READY",
                "address": public_url,
                "browser_opened": opened,
                "authorization": "CAPABILITY_URL_PRINTED_SEPARATELY",
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    print(f"Open this private local URL if no page appeared:\n{url}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return {"status": "CLOSED", "address": public_url, "summary": session.summary()}


__all__ = ["create_server", "open_local_review_page", "run_web_review"]
