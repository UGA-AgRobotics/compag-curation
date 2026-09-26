"""Stop-process ownership, original-pixel COCO masks, and browser API tests."""
from __future__ import annotations
import base64
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


s = module('start_workflow', ROOT / 'start_compag.py')
w = module('web_workflow', ROOT / 'easy_start_web.py')
c = module('coco_workflow', ROOT / 'tools/export_session_coco.py')


def wait_for(predicate, timeout=15):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.03)
    raise AssertionError('Timed out waiting for state')


class StopTests(unittest.TestCase):
    def test_stop_reaps_running_process_and_blocks_next_command(self):
        with tempfile.TemporaryDirectory() as folder:
            app = s.Launcher(ROOT / 'bundled-assets', folder, folder, log=lambda _: None)
            ready = Path(folder) / 'ready'
            errors = []
            def run():
                try:
                    app._run([sys.executable, '-c',
                              f"import pathlib,time; pathlib.Path({str(ready)!r}).touch(); time.sleep(100)"], 'test')
                except Exception as exc:
                    errors.append(exc)
            thread = threading.Thread(target=run)
            thread.start()
            wait_for(ready.exists)
            process = app.control.process
            app.control.cancel()
            thread.join(12)
            self.assertFalse(thread.is_alive())
            self.assertIsNotNone(process.poll())
            self.assertIsInstance(errors[0], s.JobStopped)
            with self.assertRaises(s.JobStopped):
                app._run([sys.executable,'-c','pass'], 'next')

    def test_stop_terminates_unresponsive_descendant(self):
        with tempfile.TemporaryDirectory() as folder:
            pidfile = Path(folder) / 'pid'
            app = s.Launcher(ROOT,folder,folder,log=lambda _:None)
            child = f"import signal,os,pathlib,time; signal.signal(signal.SIGINT,signal.SIG_IGN); pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(100)"
            parent = f"import subprocess,sys,time,signal; signal.signal(signal.SIGINT,signal.SIG_IGN); p=subprocess.Popen([sys.executable,'-c',{child!r}]); p.wait()"
            errors=[]
            def run():
                try: app._run([sys.executable,'-c',parent],'tree')
                except s.JobStopped as exc: errors.append(exc)
            thread=threading.Thread(target=run);thread.start()
            wait_for(lambda:pidfile.exists() and pidfile.read_text())
            pid=int(pidfile.read_text())
            app.control.cancel();thread.join(13)
            self.assertFalse(thread.is_alive());self.assertTrue(errors)
            def gone():
                path=Path(f'/proc/{pid}/stat')
                try:
                    return path.read_text().split()[2]=='Z'
                except (FileNotFoundError, ProcessLookupError):
                    return True
            wait_for(gone)

    def test_bridge_stop_state_and_later_job(self):
        bridge=w._Bridge(s)
        with tempfile.TemporaryDirectory() as folder:
            bridge.workspace=Path(folder);bridge.prefix=Path(folder)
            bridge.start_job('train',lambda app: app._run([sys.executable,'-c','import time;time.sleep(100)'],'training'))
            wait_for(lambda:bridge.job_control.process is not None)
            bridge.stop_job()
            wait_for(lambda:bridge.job['status']=='STOPPED')
            self.assertIsNone(bridge.job['error'])
            bridge.start_job('check',lambda app:{'status':'PASS'})
            wait_for(lambda:bridge.job['status']=='DONE')
            bridge.close()


class CocoTests(unittest.TestCase):
    def setUp(self):
        try:
            global np, cv2
            import numpy as np
            import cv2
            from compag_curation.canonical.proposals import canonical_mask_sha256
        except ImportError:
            self.skipTest('Run with the installed COMPAG Python for mask tests')
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.session=self.root/'session';self.session.mkdir()

    def decode(self,rle):
        bits=np.repeat(np.arange(len(rle['counts']))%2,rle['counts']).astype(np.uint8)
        return bits.reshape(tuple(rle['size']),order='F')

    def test_warp_rle_matches_full_image_reference_including_holes(self):
        for matrix in [np.eye(3),np.array([[.93,.02,9],[-.03,1.1,15],[.00001,.00002,1.]])]:
            for origin in [(0,0),(140,90)]:
                a=np.zeros((512,512),dtype=np.uint8);a[30:55,18:30]=1;a[35:45,22:26]=0;a[60:62,37:39]=1
                meta={'inverse_matrix':matrix.tolist(),'offset_x':origin[0],'offset_y':origin[1]}
                crop,x,y=c.source_mask(a,meta,800,700)
                actual=self.decode(c.full_rle(crop,x,y,800,700))
                transform=matrix@np.array([[1,0,origin[0]],[0,1,origin[1]],[0,0,1]])
                expected=cv2.warpPerspective(a,transform,(800,700),flags=cv2.INTER_NEAREST)
                np.testing.assert_array_equal(actual,expected)

    def fixture(self):
        from compag_curation.canonical.proposals import canonical_mask_sha256
        project=self.session/'prepared';project.mkdir();(project/'originals').mkdir()
        inference=self.session/'inference';inference.mkdir();(inference/'scores').mkdir()
        review=self.session/'review';review.mkdir()
        def write(path,value): path.write_text(json.dumps(value))
        def table(path,values):
            with path.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
        photo=project/'originals/photo.png';cv2.imwrite(str(photo),np.zeros((512,512,3),dtype=np.uint8))
        tile='photo_y00000x00000.jpg'
        original={'images':[{'id':1,'file_name':'photo.png','width':512,'height':512}],'annotations':[],'categories':[]}
        tiled={'images':[{'id':1,'file_name':tile,'width':512,'height':512,'meta':{'orig_image_id':1,'inverse_matrix':np.eye(3).tolist(),'offset_x':0,'offset_y':0}}],'annotations':[],'categories':[]}
        original['categories']=tiled['categories']=[{'id':1,'name':'CJ'},{'id':2,'name':'non-CJ'}]
        write(project/'original_coco.json',original);write(project/'tiled_coco.json',tiled)
        write(project/'PROJECT_MANIFEST.json',{'sha256_by_relative_path':{'originals/photo.png':c.sha(photo)}})
        write(project/'PROJECT_RECEIPT.json',{'project_manifest_sha256':c.sha(project/'PROJECT_MANIFEST.json')})
        proposals=[];scores=[]
        for i in range(6):
            mask=np.zeros((512,512),dtype=np.uint8);mask[20+i*15:30+i*15,30:45]=1
            proposals.append({'tile_name':tile,'proposal_index':str(i),'mask_sha256':canonical_mask_sha256(mask),
                              'mask_packbits_base64':base64.b64encode(np.packbits(mask.ravel(),bitorder='little')).decode()})
            scores.append({'image':tile,'id':str(i),'xgb_pred':'1','final_pred':'1','xgb_p':'.8'})
        table(inference/'proposals.csv',proposals);table(inference/'scores/detections.csv',scores)
        full={'schema':'compag-r92-infer-full/v1','status':'PASS','full_prepared_set':True,'expected_tile_count':1,'processed_tile_count':1,'tiles':[{'tile':tile}],
              'score_receipt':{'detections_sha256':c.sha(inference/'scores/detections.csv'),'input_sha256':'f','model_archive_sha256':'m'},
              'feature_csv_sha256':'f','r92_model_archive_sha256':'m','project_receipt_sha256':c.sha(project/'PROJECT_RECEIPT.json'),
              'original_coco_sha256':c.sha(project/'original_coco.json'),'tiled_coco_sha256':c.sha(project/'tiled_coco.json'),
              'proposal_csv_sha256':c.sha(inference/'proposals.csv'),'candidate_count':6}
        write(inference/'FULL_INFERENCE_RECEIPT.json',full)
        events=[]
        for i,(action,label) in enumerate([('accept','1'),('flip','0'),('bulk_accept','1'),('delete',''),('skip','')]):
            events.append({'image':tile,'id':str(i),'action':action,'human_label':label,'timestamp':str(i+1)})
        table(review/'review_labels.csv',events)
        write(review/'R92_REVIEW_STATE.json',{'scored_sha256':c.sha(inference/'scores/detections.csv'),'scored_csv':str(inference/'scores/detections.csv')})
        write(self.session/'EASY_SESSION.json',{'status':'SCORED','inference':str(inference),'prepared':str(project),'review_state':str(review)})
        return inference,review

    def test_archive_preserves_decisions_and_excludes_deleted_and_skipped(self):
        inference,review=self.fixture();before=c.sha(review/'review_labels.csv')
        out=self.root/'export.zip';receipt=c.export([self.session],out)
        self.assertEqual(receipt['annotation_count'],4)
        self.assertEqual(receipt['deleted'],1);self.assertEqual(receipt['skipped'],1)
        with zipfile.ZipFile(out) as z:
            data=json.loads(z.read('annotations.json'))
            self.assertEqual([a['category_id'] for a in data['annotations']],[1,2,1,1])
            self.assertEqual(data['annotations'][-1]['meta']['label_origin'],'model_prediction')
            self.assertIn(data['images'][0]['file_name'],z.namelist())
            for ann in data['annotations']:
                self.assertEqual(int(self.decode(ann['segmentation']).sum()),ann['area'])
        self.assertEqual(c.sha(review/'review_labels.csv'),before)
        only=c.export([self.session],self.root/'reviewed.zip','reviewed')
        self.assertEqual(only['annotation_count'],3)
        with self.assertRaises(FileExistsError):c.export([self.session],out)

    def test_tampered_mask_or_incomplete_inference_cannot_export(self):
        inference,_=self.fixture()
        (inference/'proposals.csv').write_text('changed')
        with self.assertRaisesRegex(ValueError,'mask table changed'):c.export([self.session],self.root/'bad.zip')
        self.assertFalse((self.root/'bad.zip').exists())
        session=json.loads((self.session/'EASY_SESSION.json').read_text());session['status']='INTERRUPTED'
        (self.session/'EASY_SESSION.json').write_text(json.dumps(session))
        with self.assertRaisesRegex(ValueError,'Finish analysis'):c.export([self.session],self.root/'bad2.zip')

    def test_export_before_review_and_empty_saved_label_scope(self):
        _,review=self.fixture()
        (review/'review_labels.csv').unlink()
        (review/'R92_REVIEW_STATE.json').unlink()
        current=c.export([self.session],self.root/'unreviewed.zip')
        self.assertEqual(current['annotation_count'],6)
        self.assertEqual(current['machine_labels'],6)
        only=c.export([self.session],self.root/'empty.zip','reviewed')
        self.assertEqual(only['annotation_count'],0)
        self.assertEqual(only['image_count'],1)


class PaperReadinessTests(unittest.TestCase):
    def test_unchanged_protocol_accepts_clean_groups_and_rejects_bulk_or_four_groups(self):
        try:
            from compag_curation.canonical.spec import CANONICAL_FEATURE_ORDER
            from compag_curation.r92_project_model import SNAPSHOT_SCHEMA
            from compag_curation.r92_sample import ARCHIVE_SHA256
        except ImportError:
            self.skipTest('Run with installed COMPAG Python for paper preflight')
        paper=module('paper_preflight',ROOT/'tools/train_paper_xgb_round.py')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);snapshot=root/'fixture.json'
            def fixture(groups=6,bulk=False):
                rows=[]
                for group in range(groups):
                    for label in [0,1]*4:
                        action='bulk_accept' if bulk else 'accept'
                        rows.append({'candidate_id':str(len(rows)),'group_id':f'photo{group}',
                                     'original_sha256':f'{group:064x}','features':[.1]*93,
                                     'label':label,'action':action,'review_factor':1.,
                                     'review_event':{'action':action,'label':label}})
                snapshot.write_text(json.dumps({'schema':SNAPSHOT_SCHEMA,'status':'PASS',
                    'review_origin':'human','software_test_fixture_only':True,
                    'native_model_sha256':ARCHIVE_SHA256,'feature_order':list(CANONICAL_FEATURE_ORDER),
                    'cumulative_eligible_count':len(rows),'rows':rows}))
            model=ROOT/'bundled-assets'/s.MODEL
            fixture()
            ready,_=paper.preflight(snapshot,model,root/'unused')
            self.assertEqual(ready['group_folds'],5)
            self.assertGreaterEqual(ready['train_card_count'],5)
            fixture(groups=4)
            with self.assertRaisesRegex(ValueError,'at least six'):paper.preflight(snapshot,model,root/'unused')
            fixture(bulk=True)
            with self.assertRaisesRegex(ValueError,'bulk-accepted'):paper.preflight(snapshot,model,root/'unused')


class WebTests(unittest.TestCase):
    def test_stop_requires_csrf_and_download_is_confined(self):
        server=w._Server(s)
        with tempfile.TemporaryDirectory() as folder:
            server.bridge.workspace=Path(folder)
            thread=threading.Thread(target=server.serve_forever);thread.start()
            root=f'http://127.0.0.1:{server.server_port}'
            base=root+'/t/'+server.token
            try:
                with self.assertRaises(urllib.error.HTTPError) as e:
                    urllib.request.urlopen(urllib.request.Request(base+'/api/stop-job',data=b'{}',headers={'Content-Type':'application/json'}))
                self.assertEqual(e.exception.code,403)
                headers={'Content-Type':'application/json','Origin':root,'X-COMPAG-CSRF':server.csrf}
                with urllib.request.urlopen(urllib.request.Request(base+'/api/stop-job',data=b'{}',headers=headers)) as r:
                    self.assertEqual(json.load(r)['status'],'IDLE')
                with self.assertRaises(urllib.error.HTTPError) as e:
                    urllib.request.urlopen(base+'/downloads/../../EASY_START_SETTINGS.json')
                self.assertEqual(e.exception.code,404)
            finally:
                server.shutdown();server.server_close();thread.join();server.bridge.close()


if __name__=='__main__':unittest.main()
