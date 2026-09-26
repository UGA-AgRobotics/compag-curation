#!/usr/bin/env python3
"""Export saved candidate masks in original-photo coordinates without training."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def rows(path, fields):
    with Path(path).open(newline='') as f:
        items = list(csv.DictReader(f))
    result = {tuple(r[k] for k in fields): r for r in items}
    if len(result) != len(items):
        raise ValueError('Duplicate candidate identity in saved inference')
    return result


def source_mask(mask, meta, width, height):
    """Warp only the foreground's source rectangle, retaining exact pixel masks."""
    import cv2
    import numpy as np
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError('An inference mask is empty')
    matrix = np.asarray(meta['inverse_matrix'], dtype=float) @ np.array(
        [[1, 0, meta['offset_x']], [0, 1, meta['offset_y']], [0, 0, 1]])
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('Invalid original-photo transform')
    corners = np.array([[[x.min()-1, y.min()-1], [x.max()+1, y.min()-1],
                         [x.max()+1, y.max()+1], [x.min()-1, y.max()+1]]], dtype=float)
    transformed = cv2.perspectiveTransform(corners, matrix)[0]
    left, top = np.maximum(np.floor(transformed.min(axis=0)).astype(int)-2, 0)
    right, bottom = np.minimum(np.ceil(transformed.max(axis=0)).astype(int)+3, [width, height])
    if right <= left or bottom <= top:
        raise ValueError('Mask is outside the original photo')
    local = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1]]) @ matrix
    crop = cv2.warpPerspective(mask, local, (int(right-left), int(bottom-top)), flags=cv2.INTER_NEAREST)
    yy, xx = np.nonzero(crop)
    if not len(xx):
        raise ValueError('Mask vanished in original coordinates')
    x0, x1, y0, y1 = int(xx.min()), int(xx.max()+1), int(yy.min()), int(yy.max()+1)
    return crop[y0:y1, x0:x1], int(left+x0), int(top+y0)


def full_rle(crop, left, top, width, height):
    """Column-major full-image RLE without allocating a full-size mask."""
    import numpy as np
    counts = [0]
    bit = 0

    def run(value, length):
        nonlocal bit
        if length <= 0:
            return
        if value == bit:
            counts[-1] += int(length)
        else:
            counts.append(int(length))
            bit = value

    run(0, left * height)
    for column in crop.T:
        run(0, top)
        changes = np.flatnonzero(column[1:] != column[:-1]) + 1
        edges = np.concatenate(([0], changes, [len(column)]))
        for begin, end in zip(edges[:-1], edges[1:]):
            run(int(column[begin]), int(end-begin))
        run(0, height-top-len(column))
    run(0, (width-left-crop.shape[1])*height)
    if sum(counts) != width*height:
        raise ValueError('COCO mask size differs from original photo')
    return {'size': [height, width], 'counts': counts}


def session_inputs(sessions):
    """Resolve all references before export. EASY_SESSION v1 uses its saved
    absolute `path` as run identity; copied records retain that identity.
    Older records without path use the resolved directory. Image/mask bytes
    are deliberately not session identities: genuine reruns stay distinct.
    """
    resolved, paths, identities = [], set(), set()
    for value in sessions:
        path = Path(value).expanduser().resolve(strict=True)
        if path in paths:
            raise ValueError('Duplicate session input: the same session was selected more than once')
        record = load(path / 'EASY_SESSION.json')
        identity = path
        if record.get('schema') == 'compag-easy-session/v1' and record.get('path'):
            saved = Path(record['path']).expanduser()
            if not saved.is_absolute():
                raise ValueError('EASY_SESSION v1 requires an absolute session path identity')
            identity = saved.resolve(strict=False)
        if identity in identities:
            raise ValueError('Duplicate session identity: copied references describe the same run')
        paths.add(path)
        identities.add(identity)
        resolved.append((path, record))
    if not resolved:
        raise ValueError('Choose at least one completed session')
    return resolved


def export(sessions, output, scope='current'):
    import numpy as np
    from compag_curation.canonical.proposals import canonical_mask_sha256
    from compag_curation.r92_project_model import _review_events
    from compag_curation.r92_review import _full_inference_binding

    if scope not in {'current', 'reviewed'}:
        raise ValueError('Choose current masks or saved labels only')
    selected = session_inputs(sessions)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new COCO output file')
    images, annotations, sources, image_files, image_ids = [], [], [], {}, {}
    totals = {'deleted': 0, 'skipped': 0, 'unreviewed_omitted': 0, 'machine_labels': 0,
              'bulk_accepted_labels': 0, 'individual_labels': 0}
    for session_path, s in selected:
        if s.get('status') not in {'SCORED', 'TRAINED'} or not s.get('inference'):
            raise ValueError('Finish analysis before exporting this photo; resume incomplete analysis first')
        inference, project = Path(s['inference']), Path(s['prepared'])
        scored = inference / 'scores/detections.csv'
        if not _full_inference_binding(scored, project / 'tiles'):
            raise ValueError('A verified full-card result is required')
        full = load(inference / 'FULL_INFERENCE_RECEIPT.json')
        project_receipt = load(project / 'PROJECT_RECEIPT.json')
        if sha(project / 'PROJECT_MANIFEST.json') != project_receipt.get('project_manifest_sha256'):
            raise ValueError('Prepared photo manifest changed since analysis')
        proposal_file = inference / 'proposals.csv'
        if sha(proposal_file) != full['proposal_csv_sha256']:
            raise ValueError('Saved proposal mask table changed')
        proposals = rows(proposal_file, ('tile_name', 'proposal_index'))
        scores = rows(scored, ('image', 'id'))
        if set(proposals) != set(scores) or len(scores) != full['candidate_count']:
            raise ValueError('Saved masks and predictions do not cover the same candidates')
        event_path = Path(s['review_state']) / 'review_labels.csv'
        before = sha(event_path) if event_path.exists() else None
        events = _review_events(event_path)
        if not set(events) <= set(scores):
            raise ValueError('Review contains a candidate outside this photo')
        if events:
            state = load(Path(s['review_state']) / 'R92_REVIEW_STATE.json')
            if state.get('scored_sha256') != sha(scored) or Path(state['scored_csv']).resolve() != scored.resolve():
                raise ValueError('Review decisions belong to another inference')
        originals = {im['id']: im for im in load(project / 'original_coco.json')['images']}
        tiles = {im['file_name']: im for im in load(project / 'tiled_coco.json')['images']}
        manifest = load(project / 'PROJECT_MANIFEST.json')['sha256_by_relative_path']
        for original in originals.values():
            name = original['file_name']
            if Path(name).name != name:
                raise ValueError('Original photo must have a simple file name')
            path = project / 'originals' / name
            identity = sha(path)
            if identity != manifest['originals/'+name]:
                raise ValueError('Original photo changed since analysis')
            if identity not in image_ids:
                image_id = len(images)+1
                archive_name = f'images/{image_id}_{name}'
                image_ids[identity] = image_id
                image_files[archive_name] = path
                images.append({'id': image_id, 'file_name': archive_name,
                               'width': original['width'], 'height': original['height']})
            original['_export_id'] = image_ids[identity]
        print(f'Exporting {session_path.name}: {len(scores)} candidates', flush=True)
        for key, proposal in sorted(proposals.items()):
            event = events.get(key)
            action = event['action'] if event else 'unreviewed'
            if action in {'delete', 'skip'}:
                totals['deleted' if action == 'delete' else 'skipped'] += 1
                continue
            if scope == 'reviewed' and not event:
                totals['unreviewed_omitted'] += 1
                continue
            score = scores[key]
            prediction = int(score.get('final_pred') or score['xgb_pred'])
            label = event['label'] if event else prediction
            if event:
                expected = 1-prediction if action in {'flip', 'sus_flip'} else prediction
                if label != expected:
                    raise ValueError('Review action differs from saved prediction')
            encoded = base64.b64decode(proposal['mask_packbits_base64'], validate=True)
            if len(encoded) != 512*512//8:
                raise ValueError('Saved mask has an invalid packed size')
            mask = np.unpackbits(np.frombuffer(encoded, dtype=np.uint8), bitorder='little').reshape(512,512)
            if canonical_mask_sha256(mask) != proposal['mask_sha256']:
                raise ValueError('Saved mask checksum differs')
            meta = tiles[key[0]]['meta']
            original = originals[meta['orig_image_id']]
            w, h = original['width'], original['height']
            crop, left, top = source_mask(mask, meta, w, h)
            origin = 'model_prediction' if not event else 'bulk_accepted_prediction' if action == 'bulk_accept' else 'individual_review'
            totals['machine_labels' if not event else 'bulk_accepted_labels' if action == 'bulk_accept' else 'individual_labels'] += 1
            annotations.append({'id': len(annotations)+1, 'image_id': original['_export_id'],
                'category_id': 1 if label == 1 else 2, 'iscrowd': 0,
                'bbox': [left, top, crop.shape[1], crop.shape[0]], 'area': int(crop.sum()),
                'segmentation': full_rle(crop,left,top,w,h),
                'meta': {'session': session_path.name, 'tile': key[0], 'proposal_index': key[1],
                         'mask_sha256': proposal['mask_sha256'], 'label_origin': origin,
                         'review_action': action, 'xgb_p': float(score['xgb_p'])}})
        if before != (sha(event_path) if event_path.exists() else None):
            raise ValueError('Review decisions changed during export; close review and try again')
        sources.append({'session': session_path.name, 'inference_sha256': sha(inference/'FULL_INFERENCE_RECEIPT.json'),
                        'review_sha256': before})
    receipt = {'schema': 'compag-saved-coco-export/v1', 'status': 'PASS', 'scope': scope,
               'image_count': len(images), 'annotation_count': len(annotations), **totals,
               'sources': sources, 'coordinates': 'original_photo_pixels',
               'overlapping_candidates': 'preserved_without_merging', 'human_ground_truth_claim': False}
    coco = {'info': {'description': 'COMPAG candidate masks; label provenance is in each annotation meta'},
            'images': images, 'annotations': annotations,
            'categories': [{'id': 1, 'name': 'CJ'}, {'id': 2, 'name': 'non-CJ'}]}
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.coco-', suffix='.tmp', dir=output.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('annotations.json', json.dumps(coco,allow_nan=False,separators=(',',':')))
            archive.writestr('EXPORT_RECEIPT.json', json.dumps(receipt,indent=2))
            for name, path in image_files.items():
                archive.write(path, name)
        # Publishing a complete archive never replaces a previous export.
        os.link(temp, output)
    finally:
        Path(temp).unlink(missing_ok=True)
    return {**receipt, 'output': str(output), 'sha256': sha(output)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', action='append', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scope', choices=['current','reviewed'], default='current')
    a = p.parse_args()
    print(json.dumps(export(a.session,a.output,a.scope)),flush=True)


if __name__ == '__main__':
    main()
