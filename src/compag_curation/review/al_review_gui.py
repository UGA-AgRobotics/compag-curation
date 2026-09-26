# al_review_gui_two_stage_clean.py
# Standalone, hard-coded GUI for human labeling on an Active-Learning SHORTLIST.
# Two-stage workflow:
#   1) pipeline writes:
#        - detections.csv           (full)
#        - al_candidates_full.csv   (full-schema shortlist: Top-K rows)
#   2) This GUI reads al_candidates_full.csv so you ONLY review the shortlist,
#      but you can press 'v' to toggle into full view for spot-checking.
#
# Keys:
#   Labeling:
#     Right click / 'a' / '1' : accept (set human_label = model_pred)
#     Left  click / 'r' / '2' : flip   (set human_label = 1 - model_pred)
#     's' / Enter / '3'       : skip (no label)
#     'p' / 'n'               : prev / next
#     'z' / '0'               : undo last saved label
#     'q'                     : quit
#   View:
#     'v' : toggle shortlist <-> full
#     'f' : (only in full view) cycle full filter: all -> uncertain -> certain
#     'h' : toggle help lines (title always shown)

from __future__ import annotations
import io, json, time, re
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

from compag_curation.contracts import FileIdentity
from compag_curation.review.adapters import ReviewTableStore

# ============================= CONSTANTS (edit here) =============================
START_VIEW      = "short"   # "short" | "full"

# Model decision params (should match pipeline)
USE_XGB  = True
USE_YOLO = False

DET_POLICY        = "xgb"     # "xgb" | "hybrid" | "yolo" | "and" | "or"
DET_MISSING       = "reject"  # for hybrid/and/or when one source is missing: "reject"|"ignore"
THR_XGB           = 0.40
THR_YOLO          = 0.20
THR_IOU           = 0.60
DET_THR           = 0.50
HYBRID_YOLO_BIAS  = 0.80

# Active learning uncertainty band
AL_MARGIN         = 0.20     # |p - thr| <= margin considered "uncertain"

# UI
COLOR_BY          = "final"  # "final" | "xgb" | "kept"
HEADER_H          = 90
CJ_NAME, NONCJ_NAME = "CJ", "Non-CJ"

GREEN = (0,255,0); RED = (0,0,255); WHITE = (255,255,255); BLACK = (0,0,0); YELLOW = (0,255,255)

# ================================================================================

# ---------- Small UI helpers ----------
def draw_text(img, text, org=(10,25), color=WHITE, scale=0.7, thick=2, bg=None):
    if bg is not None:
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x, y = org
        cv2.rectangle(img, (x-4, y-th-6), (x+tw+4, y+6), bg, -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def build_canvas(base_img, title_lines, show_help=True):
    """Title line always shown; help lines are toggleable."""
    h,w=base_img.shape[:2]
    header=np.zeros((HEADER_H,w,3),dtype=np.uint8)
    y=24
    if title_lines:
        draw_text(header, title_lines[0], (10,y), WHITE, 0.7, 2); y+=26
    if show_help and len(title_lines)>1:
        for line in title_lines[1:]:
            draw_text(header, line, (10,y), WHITE, 0.7, 2); y+=26
    return np.vstack([header,base_img])

def load_image(root:Path, img_name:str):
    p=Path(img_name)
    if not p.is_absolute():
        p=root / img_name
    im=cv2.imread(str(p))
    return im,p

def _safe_float(v, default=0.0)->float:
    try:
        if v is None or (isinstance(v,float) and np.isnan(v)): return float(default)
        return float(v)
    except Exception:
        try: return float(str(v).strip())
        except Exception: return float(default)

def parse_bbox(row):
    def _getf(k, default=None):
        if k in row:
            try: return float(row[k])
            except Exception:
                try: return float(str(row[k]).strip())
                except Exception: return default
        return default
    def _ints(*vals): return tuple(int(round(v)) for v in vals)

    if all(k in row for k in ("bbox_x","bbox_y","bbox_w","bbox_h")):
        return _ints(_getf("bbox_x",0),_getf("bbox_y",0),_getf("bbox_w",0),_getf("bbox_h",0))
    if all(k in row for k in ("x","y","w","h")):
        return _ints(_getf("x",0),_getf("y",0),_getf("w",0),_getf("h",0))
    if "bbox" in row and str(row["bbox"]).strip():
        s=str(row["bbox"]).strip()
        if "," in s and not s.startswith("["):
            try:
                x,y,w,h=[float(t) for t in s.split(",")]
                return _ints(x,y,w,h)
            except Exception:
                pass
        try:
            bb=json.loads(s)
            if isinstance(bb,(list,tuple)) and len(bb)>=4:
                return _ints(bb[0],bb[1],bb[2],bb[3])
        except Exception:
            pass

    # fallback: from poly
    if "poly" in row and row["poly"] not in (None,""):
        try:
            vals=row["poly"]
            if isinstance(vals,str): vals=json.loads(vals)
            if isinstance(vals,(list,tuple)) and len(vals)>=4:
                xs=[float(x) for x in vals[0::2]]; ys=[float(y) for y in vals[1::2]]
                xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys)
                x=int(round(xmin)); y=int(round(ymin))
                w=int(round(xmax-xmin))+1; h=int(round(ymax-ymin))+1
                return _ints(x,y,max(1,w),max(1,h))
        except Exception:
            pass

    cx=_getf("cx",_getf("centroid_x",50)); cy=_getf("cy",_getf("centroid_y",50))
    cx=50.0 if cx is None else cx; cy=50.0 if cy is None else cy
    return _ints(cx-15, cy-15, 30, 30)

def draw_detection(img, row, show_mask=True, show_bbox=True, show_id=True, color=WHITE):
    if show_mask:
        poly=row.get("poly") or row.get("polygon") or row.get("poly_str")
        if isinstance(poly,str):
            try:
                arr=json.loads(poly)
                pts=np.array(arr,dtype=np.int32).reshape(-1,2)
                overlay=img.copy()
                cv2.fillPoly(overlay,[pts],(255,255,255))
                cv2.addWeighted(overlay,0.18,img,0.82,0,img)
                cv2.polylines(img,[pts],isClosed=True,color=WHITE,thickness=2)
            except Exception:
                pass
    if show_bbox:
        x,y,w,h=parse_bbox(row)
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        if show_id:
            top=max(16,y-8)
            draw_text(img, f"id={row.get('id','?')}", (x, top), color, 0.6, 2, bg=(0,0,0))

# ---------- Decision / uncertainty ----------
def _combine_default(x_ok,y_ok,x_has,y_has,pol:str,miss:str)->bool:
    present=[]
    if x_has: present.append(bool(x_ok))
    if y_has: present.append(bool(y_ok))
    if not present: return False
    pol=pol.lower(); miss=miss.lower()
    if pol in ("xgb","xgb_only"):   return bool(x_ok) if x_has else False
    if pol in ("yolo","yolo_only"): return bool(y_ok) if y_has else False
    if pol=="or":                   return (bool(x_ok) or bool(y_ok)) if miss=="reject" else any(present)
    return (bool(x_ok) and bool(y_ok)) if miss=="reject" else all(present)

def compute_final_pred(row:dict, policy:str, missing:str,
                       thr_xgb:float, thr_yolo:float, thr_iou:float,
                       det_thr:float, hybrid_yolo_bias:float,
                       use_xgb:bool=True, use_yolo:bool=False)->tuple[int, float|None]:
    xgb_p=_safe_float(row.get("xgb_p", row.get("xgb_score", row.get("prob", 0.0))), 0.0) if use_xgb else 0.0
    y_conf=_safe_float(row.get("yolo_conf", 0.0), 0.0) if use_yolo else 0.0
    y_iou=_safe_float(row.get("yolo_iou", 0.0), 0.0) if use_yolo else 0.0

    x_has=use_xgb
    y_has=(use_yolo and y_conf>0.0)
    x_ok=(xgb_p>=float(thr_xgb)) if x_has else False
    y_ok=(y_conf>=float(thr_yolo) and y_iou>=float(thr_iou)) if y_has else False

    if policy.lower()=="hybrid" and not y_ok:
        if str(missing).lower()=="reject":
            return 0, None
        return (1 if x_ok else 0), xgb_p

    pol=policy.lower()
    if pol=="hybrid":
        def _norm(a,b):
            s=float(a)+float(b)
            return (float(a)/s, float(b)/s) if s>0 else (0.5,0.5)
        p_y=float(y_conf); x=float(xgb_p)
        if (x>=det_thr) and (p_y>=det_thr):
            wy,_=_norm(p_y,x); p_f=wy*p_y+(1.0-wy)*x; return 1,p_f
        elif (p_y>=det_thr) and (x<det_thr):
            wy,_=_norm(p_y,x); p_f=wy*p_y+(1.0-wy)*x; return (1 if p_f>=det_thr else 0),p_f
        elif (x>=det_thr) and (p_y<det_thr):
            wy_raw,_=_norm(p_y,x); wy=max(float(hybrid_yolo_bias), wy_raw); p_f=wy*p_y+(1.0-wy)*x
            return (1 if p_f>=det_thr else 0),p_f
        else:
            wy,_=_norm(p_y,x); p_f=wy*p_y+(1.0-wy)*x; return 0,p_f

    keep=_combine_default(x_ok,y_ok,x_has,y_has,pol,missing)
    return (1 if keep else 0), xgb_p

def thr_for_row(row:dict, policy:str)->float:
    pol=str(policy).lower()
    if pol=="yolo":
        return float(THR_YOLO)
    if pol=="hybrid":
        y_conf=_safe_float(row.get("yolo_conf",0.0),0.0)
        y_iou=_safe_float(row.get("yolo_iou",0.0),0.0)
        y_ok=(USE_YOLO and (y_conf>0.0) and (y_conf>=float(THR_YOLO)) and (y_iou>=float(THR_IOU)))
        return float(DET_THR if y_ok else THR_XGB)
    return float(THR_XGB)

def is_uncertain_row(row:dict, p_ui:float|None, policy:str)->bool:
    if p_ui is None:
        p_ui=_safe_float(row.get("xgb_p", row.get("xgb_score", row.get("prob",0.0))), 0.0)
    thr=thr_for_row(row, policy)
    return abs(float(p_ui)-float(thr)) <= float(AL_MARGIN)

# -------- neighbors (optional) --------
_TILE_PAT = re.compile(r"^(?P<base>.+?)_y(?P<y>\d{1,8})x(?P<x>\d{1,8})\.(?P<ext>[^.]+)$", re.IGNORECASE)
def parse_tile_name(name:str):
    m=_TILE_PAT.match(Path(name).name)
    return (m.group("base"), int(m.group("y")), int(m.group("x")), m.group("ext")) if m else None
def make_tile_name(base:str,y:int,x:int,ext:str)->str: return f"{base}_y{y:05d}x{x:05d}.{ext}"
def try_load_neighbor(img_root:Path,center_name:str,step_h:int,step_w:int,direction:str):
    parsed=parse_tile_name(center_name)
    if not parsed: return None
    base,y,x,ext=parsed
    if direction=='L': nx,ny=x-step_w,y
    elif direction=='R': nx,ny=x+step_w,y
    elif direction=='U': nx,ny=x,y-step_h
    elif direction=='D': nx,ny=x,y+step_h
    else: return None
    p=img_root/make_tile_name(base,ny,nx,ext)
    if not p.exists(): return None
    im=cv2.imread(str(p))
    return im
def compose_with_neighbors(center_img, neighbors:dict[str,np.ndarray|None]):
    H,W=center_img.shape[:2]
    def _fit(im):
        if im is None:
            b=np.zeros_like(center_img); b[:]=(32,32,32)
            cv2.putText(b,"?",(b.shape[1]//2-10, b.shape[0]//2+10),
                        cv2.FONT_HERSHEY_SIMPLEX,1.2,(96,96,96),2,cv2.LINE_AA)
            return b
        if im.shape[0]!=H or im.shape[1]!=W:
            im=cv2.resize(im,(W,H),interpolation=cv2.INTER_NEAREST)
        return im
    left=_fit(neighbors.get('L')); right=_fit(neighbors.get('R')); up=_fit(neighbors.get('U')); down=_fit(neighbors.get('D'))
    top=np.hstack([_fit(None),up,_fit(None)])
    mid=np.hstack([left,center_img,right])
    bot=np.hstack([_fit(None),down,_fit(None)])
    return np.vstack([top,mid,bot])
def arrow_direction_from_key(k_raw:int,k_small:int)->str|None:
    if k_small in (81,82,83,84): return {81:'L',82:'U',83:'R',84:'D'}[k_small]
    if k_raw   in (2424832,2490368,2555904,2621440): return {2424832:'L',2490368:'U',2555904:'R',2621440:'D'}[k_raw]
    return None

def main(
    *,
    shortlist_path: Path,
    full_detections_path: Path | None,
    images_root: Path,
    output_csv: Path,
    output_identity: FileIdentity,
    table_store: ReviewTableStore,
) -> FileIdentity:
    # Load shortlist
    short_p=Path(shortlist_path)
    if not short_p.exists():
        raise SystemExit(f"SHORTLIST_PATH not found: {short_p}")
    df_short=pd.read_csv(short_p)
    if "image" not in df_short.columns or "id" not in df_short.columns:
        raise SystemExit("Shortlist must contain columns: image, id")

    # Load full (optional)
    df_full=None
    full_p=Path(full_detections_path) if full_detections_path else None
    if full_p is not None and full_p.exists():
        df_full=pd.read_csv(full_p)
        if "image" not in df_full.columns or "id" not in df_full.columns:
            print("[WARN] Full detections missing required columns (image,id). Disabling full view.")
            df_full=None

    print(f"[LOAD] shortlist rows: {len(df_short)}")
    if df_full is not None:
        print(f"[LOAD] full rows: {len(df_full)}")
    else:
        print("[LOAD] full view: disabled")

    # Basic normalization (avoid NaNs)
    for df in [df_short] + ([df_full] if df_full is not None else []):
        if df is None: continue
        for c in ["xgb_p","yolo_conf","yolo_iou"]:
            if c in df.columns:
                df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0.0)
        if "kept" in df.columns:
            df["kept"]=pd.to_numeric(df["kept"],errors="coerce").fillna(-1).astype(int)
        else:
            df["kept"]=-1

    # Full-view filter state
    view_mode=str(START_VIEW).lower().strip()
    if view_mode not in ("short","full"):
        view_mode="short"
    if view_mode=="full" and df_full is None:
        view_mode="short"

    full_filter_mode="all"  # all | uncertain | certain
    df_full_view=df_full
    unc_cache: dict[tuple, np.ndarray] = {}

    def unc_cache_key():
        return (
            str(DET_POLICY).lower(), str(DET_MISSING).lower(),
            bool(USE_XGB), bool(USE_YOLO),
            float(THR_XGB), float(THR_YOLO), float(THR_IOU),
            float(DET_THR), float(HYBRID_YOLO_BIAS), float(AL_MARGIN),
        )

    def get_unc_mask_full():
        if df_full is None:
            return None
        key=unc_cache_key()
        if key in unc_cache:
            return unc_cache[key]
        print(f"[FULL] computing uncertainty mask for {len(df_full)} rows ...")
        def _one(r):
            d=r.to_dict()
            fp=compute_final_pred(d, DET_POLICY, DET_MISSING, THR_XGB, THR_YOLO, THR_IOU, DET_THR, HYBRID_YOLO_BIAS, USE_XGB, USE_YOLO)
            p_ui=(None if fp[1] is None else float(fp[1]))
            return bool(is_uncertain_row(d, p_ui, DET_POLICY))
        mask=df_full.apply(_one, axis=1).to_numpy(dtype=bool)
        unc_cache[key]=mask
        return mask

    def rebuild_full_view(keep_key=None):
        nonlocal df_full_view
        if df_full is None:
            df_full_view=None
            return
        if full_filter_mode=="all":
            df_full_view=df_full
            return
        mask=get_unc_mask_full()
        if mask is None:
            df_full_view=df_full
        elif full_filter_mode=="uncertain":
            df_full_view=df_full.loc[mask].reset_index(drop=True)
        else:
            df_full_view=df_full.loc[~mask].reset_index(drop=True)

    rebuild_full_view()

    # Choose active df
    df_active = df_short if view_mode=="short" else (df_full_view if df_full_view is not None else df_short)

    # Output labels file
    out_path=Path(output_csv)
    table_snapshot=table_store.read(out_path,output_identity)
    reviewed=pd.read_csv(io.BytesIO(table_snapshot.payload))
    current_output_identity=table_snapshot.identity

    def save_reviewed() -> None:
        nonlocal current_output_identity
        destination=io.StringIO(newline="")
        reviewed.to_csv(destination,index=False,lineterminator="\n")
        table_snapshot=table_store.replace(
            out_path,
            current_output_identity,
            destination.getvalue().encode("utf-8"),
        )
        current_output_identity=table_snapshot.identity

    if len(df_active)==0:
        print("[INFO] nothing to review."); return current_output_identity

    # UI state
    img_root=Path(images_root)
    show_help=True; show_overlay=True; show_mask=True; show_box=True; show_id=True
    neighbor_imgs={'L':None,'R':None,'U':None,'D':None}
    state={"label":None,"click":None}

    cv2.namedWindow("review", cv2.WINDOW_NORMAL)

    def make_mouse_cb(header_h=HEADER_H):
        def on_mouse(event,x,y,flags,userdata):
            if y < header_h: return
            y_img=y-header_h
            if event==cv2.EVENT_RBUTTONDOWN:
                state["label"]=int(userdata["model_pred"]); state["click"]=(x,y_img); userdata["action"]="accept"
            elif event==cv2.EVENT_LBUTTONDOWN:
                state["label"]=1-int(userdata["model_pred"]); state["click"]=(x,y_img); userdata["action"]="flip"
        return on_mouse

    i=0
    total=len(df_active)
    while 0<=i<total:
        row=df_active.iloc[i].to_dict()

        img,_=load_image(img_root,row["image"])
        if img is None:
            print(f"[WARN] cannot load image: {row['image']}"); i+=1; continue

        fp=compute_final_pred(row, DET_POLICY, DET_MISSING, THR_XGB, THR_YOLO, THR_IOU, DET_THR, HYBRID_YOLO_BIAS, USE_XGB, USE_YOLO)
        pred=int(fp[0])
        p_val=(fp[1] if fp[1] is not None else row.get("xgb_p",0.5))
        p=float(_safe_float(p_val,0.0))
        xgb_p=float(_safe_float(row.get("xgb_p", row.get("xgb_score", row.get("prob",0.0))), 0.0))
        xgb_pred=int(xgb_p>=float(THR_XGB))
        kept=int(row.get("kept",-1))
        unc=is_uncertain_row(row, (None if fp[1] is None else float(fp[1])), DET_POLICY)

        # draw color
        if COLOR_BY=="final": color_flag=pred
        elif COLOR_BY=="xgb": color_flag=xgb_pred
        else: color_flag=kept if kept in (0,1) else 0
        base_color=GREEN if color_flag==1 else RED
        draw_color=YELLOW if unc else base_color

        overlay=img.copy()
        if show_overlay:
            draw_detection(overlay,row,show_mask=show_mask,show_bbox=show_box,show_id=show_id,color=draw_color)

        step_h,step_w=overlay.shape[:2]

        def make_view():
            montage=compose_with_neighbors(overlay, neighbor_imgs)
            title=f"[{i+1}/{total}] view={view_mode}" + (f"/{full_filter_mode}" if view_mode=="full" else "")
            title += f" | {Path(row['image']).name} | id={row.get('id','?')} | pred={'CJ' if pred==1 else 'Non-CJ'} p={p:.3f}"
            if unc: title += " | UNCERTAIN"
            title += f" | xgb={xgb_p:.3f}"
            help1="Mouse: Right=accept Left=flip | Keys: n=next p=prev z=undo s=skip q=quit"
            help2="View: v=toggle short/full  f=filter(full)  h=help | Toggles: o/m/b/i overlay/mask/bbox/id"
            return build_canvas(montage,[title,help1,help2],show_help=show_help)

        userdata={"model_pred":pred,"action":None}
        cv2.setMouseCallback("review", make_mouse_cb(), userdata)

        decided=None  # True/False/"prev"/"undo"/"jump"
        while True:
            cv2.imshow("review", make_view())
            k_raw=cv2.waitKey(30)
            k=(-1 if k_raw==-1 else (k_raw & 0xFF))

            # quit/nav
            if k==ord('q'):
                decided=False
                break
            if k in (ord('s'), ord('3'), 13):
                decided=True; userdata["action"]="skip"
                state["label"]=None; state["click"]=None
                break
            if k==ord('n'):
                decided=True; userdata["action"]=userdata.get("action") or "skip"
                break
            if k==ord('p'):
                decided="prev"
                break
            if k in (ord('z'), ord('0')):
                decided="undo"
                break

            # quick label
            if k in (ord('a'), ord('1')):
                state["label"]=int(userdata["model_pred"]); state["click"]=(10,10); userdata["action"]="accept"
                decided=True; break
            if k in (ord('r'), ord('2')):
                state["label"]=1-int(userdata["model_pred"]); state["click"]=(10,10); userdata["action"]="flip"
                decided=True; break

            # toggles
            if k==ord('h'):
                show_help=not show_help
            elif k==ord('o'):
                show_overlay=not show_overlay
                overlay=img.copy()
                if show_overlay:
                    draw_detection(overlay,row,show_mask=show_mask,show_bbox=show_box,show_id=show_id,color=draw_color)
            elif k==ord('m'):
                show_mask=not show_mask
                overlay=img.copy()
                if show_overlay:
                    draw_detection(overlay,row,show_mask=show_mask,show_bbox=show_box,show_id=show_id,color=draw_color)
            elif k==ord('b'):
                show_box=not show_box
                overlay=img.copy()
                if show_overlay:
                    draw_detection(overlay,row,show_mask=show_mask,show_bbox=show_box,show_id=show_id,color=draw_color)
            elif k==ord('i'):
                show_id=not show_id
                overlay=img.copy()
                if show_overlay:
                    draw_detection(overlay,row,show_mask=show_mask,show_bbox=show_box,show_id=show_id,color=draw_color)

            elif k==ord('v'):
                if df_full is None:
                    print("[VIEW] full view disabled (FULL_DETECTIONS_PATH missing).")
                    continue
                cur_key=(row.get("image"), row.get("id"))
                # toggle
                view_mode="full" if view_mode=="short" else "short"
                if view_mode=="full":
                    rebuild_full_view()
                    df_active=df_full_view if df_full_view is not None else df_short
                else:
                    df_active=df_short
                total=len(df_active)
                # keep same key if possible
                try:
                    m=df_active.index[(df_active["image"]==cur_key[0]) & (df_active["id"]==cur_key[1])]
                    i=int(m[0]) if len(m) else 0
                except Exception:
                    i=0
                neighbor_imgs={'L':None,'R':None,'U':None,'D':None}
                decided="jump"
                break

            elif k==ord('f'):
                if view_mode!="full":
                    print("[FILTER] press 'v' to enter full view first.")
                    continue
                if df_full is None:
                    print("[FILTER] full view disabled.")
                    continue
                cur_key=(row.get("image"), row.get("id"))
                order=["all","uncertain","certain"]
                full_filter_mode = order[(order.index(full_filter_mode)+1) % len(order)]
                print(f"[FILTER] full_filter_mode -> {full_filter_mode}")
                rebuild_full_view()
                df_active=df_full_view if df_full_view is not None else df_short
                total=len(df_active)
                # keep key
                try:
                    m=df_active.index[(df_active["image"]==cur_key[0]) & (df_active["id"]==cur_key[1])]
                    i=int(m[0]) if len(m) else 0
                except Exception:
                    i=0
                neighbor_imgs={'L':None,'R':None,'U':None,'D':None}
                decided="jump"
                break

            # neighbor arrows
            dir_key=arrow_direction_from_key(k_raw,k)
            if dir_key is not None:
                if neighbor_imgs.get(dir_key) is not None:
                    neighbor_imgs[dir_key]=None
                else:
                    neighbor_imgs[dir_key]=try_load_neighbor(img_root,row["image"],step_h,step_w,dir_key)

            # mouse decided?
            if state["label"] is not None and state["click"] is not None:
                decided=True
                break

        if decided is False:
            break

        if decided=="jump":
            continue

        if decided=="prev":
            i=max(0,i-1)
            state["label"]=None; state["click"]=None
            neighbor_imgs={'L':None,'R':None,'U':None,'D':None}
            continue

        if decided=="undo":
            if len(reviewed):
                last=(reviewed.tail(1)["image"].iloc[0], reviewed.tail(1)["id"].iloc[0])
                reviewed=reviewed.iloc[:-1].reset_index(drop=True)
                save_reviewed()
                # go back to that item if exists
                try:
                    m=df_active.index[(df_active["image"]==last[0]) & (df_active["id"]==last[1])]
                    i=int(m[0]) if len(m) else max(0,i-1)
                except Exception:
                    i=max(0,i-1)
                print("[UNDO] removed last label.")
            state["label"]=None; state["click"]=None
            neighbor_imgs={'L':None,'R':None,'U':None,'D':None}
            continue

        # decided True: write if accept/flip
        if userdata.get("action") in ("accept","flip") and state["label"] is not None:
            human=int(state["label"])
            cx,cy=state["click"] if state["click"] is not None else (-1,-1)
            rec={
                "image":row["image"], "id":row["id"],
                "human_label":human, "action":userdata["action"],
                "click_x":cx, "click_y":cy,
                "prob":p, "xgb_prob":float(xgb_p),
                "final_pred":pred, "xgb_pred":int(xgb_pred),
                "kept":kept,
                "yolo_conf":float(row.get("yolo_conf",0.0)), "yolo_iou":float(row.get("yolo_iou",0.0)),
                "det_policy":DET_POLICY, "det_missing":DET_MISSING,
                "thr_xgb":float(THR_XGB), "thr_yolo":float(THR_YOLO), "thr_iou":float(THR_IOU),
                "det_thr":float(DET_THR), "hybrid_yolo_bias":float(HYBRID_YOLO_BIAS),
                "use_xgb":bool(USE_XGB), "use_yolo":bool(USE_YOLO),
                "timestamp":int(time.time())
            }
            reviewed=pd.concat([reviewed,pd.DataFrame([rec])],ignore_index=True)
            save_reviewed()

        # next
        i+=1
        state["label"]=None; state["click"]=None
        neighbor_imgs={'L':None,'R':None,'U':None,'D':None}

    cv2.destroyAllWindows()
    print(f"[DONE] saved reviews to: {out_path}")
    return current_output_identity

if __name__=="__main__":
    raise SystemExit("Use the typed compag-curation review handler")
