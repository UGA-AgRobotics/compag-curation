"""R5.4 owned-process lifecycle and duplicate-session regression contracts.

Run with the installed COMPAG Python; all labels/masks here are software fixtures.
"""
from pathlib import Path
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('repair_fixtures', ROOT/'tools/test_easy_start_workflow.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
s, w, c = f.s, f.w, f.c


def alive(pid):
    try:
        return s.JobControl._stat(pid)[0] not in {'Z', 'X'}
    except FileNotFoundError:
        return False


class StopRepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.b = w._Bridge(s)
        self.b.workspace = self.root
        self.b.prefix = self.root
        self.addCleanup(self.b.close)

    def test_closed_stdio_descendant_twenty_synchronized_runs(self):
        # The child's ready receipt follows installation of its SIGINT handler.
        # Only escalation delays are shortened; the production cleanup path runs.
        with mock.patch.object(s.JobControl, 'INTERRUPT_GRACE', .15):
            for run in range(20):
                with self.subTest(run=run):
                    ready = self.root/f'child-{run}'
                    child = f'import signal,os,time;from pathlib import Path;signal.signal(signal.SIGINT,signal.SIG_IGN);Path({str(ready)!r}).write_text(str(os.getpid()));time.sleep(60)'
                    parent = f'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",{child!r}],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);time.sleep(60)'
                    self.b.start_job('train', lambda app: app._run([sys.executable,'-c',parent],'fixture'))
                    f.wait_for(lambda: ready.exists() and ready.read_text())
                    pid = int(ready.read_text())
                    control = self.b.job_control
                    self.b.stop_job(self.b.job['id']); self.b.stop_job(self.b.job['id'])
                    f.wait_for(lambda: self.b.job['status'] == 'STOPPED')
                    self.assertFalse(alive(pid), 'STOPPED must never coexist with live owned work')
                    self.assertTrue(control.cleanup_done.is_set())
                    self.assertIsNone(control.process)
                    self.assertFalse(control._cleanup_thread.is_alive())

    def test_pending_cleanup_gates_next_job_and_stale_stop_cannot_cancel_next(self):
        entered, release = threading.Event(), threading.Event()
        original = s.JobControl._cleanup
        def held(control, process):
            entered.set(); release.wait(5); original(control, process)
        with mock.patch.object(s.JobControl, '_cleanup', held):
            first = self.b.start_job('train', lambda app: app._run([sys.executable,'-c','import time;time.sleep(60)'],'fixture'))
            f.wait_for(lambda: self.b.job_control.process is not None)
            self.b.stop_job(first['job_id']); self.assertTrue(entered.wait(3))
            try:
                self.assertEqual(self.b.job['status'], 'STOPPING')
                self.assertIsNotNone(self.b.job_control.process)
                with self.assertRaises(s.StartError): self.b.start_job('train', lambda app: {})
            finally: release.set()
            f.wait_for(lambda: self.b.job['status'] == 'STOPPED')
        done = threading.Event()
        self.b.start_job('train', lambda app: done.wait(3))
        try:
            with self.assertRaisesRegex(s.StartError,'older job'): self.b.stop_job(first['job_id'])
            self.assertFalse(self.b.job_control.stopped.is_set())
        finally: done.set()
        f.wait_for(lambda: self.b.job['status'] == 'DONE')

    def test_stop_during_spawn_is_serialized(self):
        entered, release = threading.Event(), threading.Event()
        original = subprocess.Popen
        def spawn(*args, **kwargs):
            p = original(*args, **kwargs); entered.set(); release.wait(5); return p
        with mock.patch.object(s.subprocess, 'Popen', spawn):
            self.b.start_job('train', lambda app: app._run([sys.executable,'-c','import time;time.sleep(60)'],'fixture'))
            self.assertTrue(entered.wait(3))
            stopper = threading.Thread(target=self.b.stop_job); stopper.start(); release.set(); stopper.join(3)
        f.wait_for(lambda: self.b.job['status'] == 'STOPPED')
        self.assertIsNone(self.b.job_control.process)

    def test_cleanup_failure_blocks_retry_without_signaling_reused_identity(self):
        c1 = s.JobControl(); p = c1.spawn([sys.executable,'-c','import time;time.sleep(60)'])
        try:
            c1._leader_start += 1  # Simulate a mismatched start-time identity.
            with mock.patch.object(s.os, 'killpg') as kill:
                c1.cancel(); self.assertTrue(c1.cleanup_done.wait(3))
                with self.assertRaises(s.JobCleanupError): c1.finished(p)
                kill.assert_not_called()
                self.assertIs(c1.process, p)
            self.b._app = lambda log=None: type('App', (), {'control':c1})()
            self.b.start_job('train', lambda app: app.control.check())
            f.wait_for(lambda: self.b.job['status'] == 'CLEANUP_FAILED')
            with self.assertRaises(s.StartError): self.b.start_job('train', lambda app: {})
        finally:
            # The fixture holds the exact unreaped Popen leader it created.
            p.terminate(); p.wait(); c1.process = None

    def test_worker_failure_cleans_owned_work_and_allows_retry(self):
        child = []
        def failed(app):
            child.append(app.control.spawn([sys.executable,'-c','import time;time.sleep(60)']))
            raise ValueError('controlled failure')
        self.b.start_job('train', failed)
        f.wait_for(lambda: self.b.job['status'] == 'ERROR')
        self.assertIsNotNone(child[0].returncode)
        self.b.start_job('train', lambda app: {'status':'PASS'})
        f.wait_for(lambda: self.b.job['status'] == 'DONE')

    def test_completed_output_review_and_unrelated_process_survive_stop(self):
        review = self.root/'review.csv'; review.write_text('immutable fixture decisions\n')
        artifact = self.root/'completed'; committed, release = threading.Event(), threading.Event()
        sentinel = subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
        try:
            def run(app):
                app._run([sys.executable,'-c',f'from pathlib import Path;Path({str(artifact)!r}).write_text("complete")'],'fixture')
                committed.set(); release.wait(3); return {'status':'PASS'}
            self.b.start_job('train',run); self.assertTrue(committed.wait(3))
            self.b.stop_job(); release.set(); f.wait_for(lambda:self.b.job['status']=='DONE')
            self.assertEqual(artifact.read_text(),'complete')
            self.assertEqual(review.read_text(),'immutable fixture decisions\n')
            self.assertIsNone(sentinel.poll())
        finally: release.set(); sentinel.terminate(); sentinel.wait()


class CocoRepairTests(unittest.TestCase):
    def setUp(self):
        self.f = f.CocoTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.f.fixture(); self.root, self.session = self.f.root, self.f.session

    def test_one_session_four_and_duplicate_rejected_before_output(self):
        self.assertEqual(c.export([self.session],self.root/'single.zip')['annotation_count'],4)
        with mock.patch.object(c.tempfile,'mkstemp') as create:
            with self.assertRaisesRegex(ValueError,'Duplicate session'): c.export([self.session,self.session],self.root/'new/output.zip')
            create.assert_not_called()
        self.assertFalse((self.root/'new').exists())

    def test_absolute_relative_dotdot_symlink_aliases_rejected(self):
        alias = self.root/'alias'; alias.symlink_to(self.session,target_is_directory=True)
        variants = [Path(os.path.relpath(self.session)),self.session/'..'/'session',alias]
        for alias in variants:
            with self.subTest(alias=str(alias)), self.assertRaisesRegex(ValueError,'Duplicate session'):
                c.export([self.session,alias],self.root/'none.zip')
            self.assertFalse((self.root/'none.zip').exists())

    def test_copied_v1_identity_rejected_distinct_runs_preserved(self):
        p=self.session/'EASY_SESSION.json'; record=json.loads(p.read_text())
        record.update(schema='compag-easy-session/v1',path=str(self.session));p.write_text(json.dumps(record))
        copied=self.root/'copy';copied.mkdir();q=copied/'EASY_SESSION.json';q.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'Duplicate session identity'): c.export([self.session,copied],self.root/'bad.zip')
        self.assertFalse((self.root/'bad.zip').exists())
        record['path']=str(copied);q.write_text(json.dumps(record))
        result=c.export([self.session,copied],self.root/'distinct.zip')
        self.assertEqual((result['image_count'],result['annotation_count']),(1,8))


if __name__ == '__main__': unittest.main(verbosity=2)
