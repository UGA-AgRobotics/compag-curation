"""Local launcher regression checks; no model fitting or GPU is required."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import shutil
import subprocess
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s = load("start_compag")
web = load("easy_start_web")


class YoloLocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "releases" / "compag"
        self.repo.mkdir(parents=True)
        (self.repo / "tools").mkdir()
        (self.repo / "tools/setup_yolo_segment_env.py").touch()
        self.patch_root = patch.object(s, "ROOT", self.repo)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)
        self.app = s.Launcher(self.repo / "bundled-assets", self.root / "env",
                              self.root / "workspace", log=lambda message: None)

    def package(self, path=None, content=b"synthetic detector A"):
        path = path or self.root / "model-package"
        (path / "weights").mkdir(parents=True)
        (path / "weights/best.pt").write_bytes(content)
        sha = s._sha256(path / "weights/best.pt")
        s._write_json(path / "MODEL_MANIFEST.json", {
            "checkpoint_sha256": sha, "task": "instance segmentation",
            "class_mapping": {"0": "CJ"}})
        return path, sha

    def test_first_use_finds_nearby_package_with_no_paths(self):
        path, sha = self.package()
        result = self.app.yolo_locations({})
        self.assertEqual(result["model_package"], str(path))
        self.assertEqual(result["checkpoint_sha256"], sha)
        self.assertEqual(result["status"], "FOUND")
        self.assertFalse(self.app.settings_path.exists())

    def test_remembers_external_location_after_yolo_off(self):
        path, sha = self.package(self.root / "outside" / "private-package")
        result = self.app.yolo_locations({"latest_round": {"yolo_mode": "off"},
            "yolo_defaults": {"model_package": str(path), "checkpoint_sha256": sha}})
        self.assertEqual(result["model_package"], str(path))
        self.assertEqual(result["source"], "saved")

    def test_relocated_package_must_have_saved_checkpoint_identity(self):
        path, sha = self.package()
        self.package(self.root / "wrong-model", b"different detector")
        saved = {"yolo_defaults": {"model_package": str(self.root / "old-path"),
                                   "checkpoint_sha256": sha}}
        self.assertEqual(self.app.yolo_locations(saved)["model_package"], str(path))
        (path / "weights/best.pt").unlink()
        self.assertEqual(self.app.yolo_locations(saved)["status"], "MISSING")

    def test_multiple_distinct_models_require_one_explicit_selection(self):
        self.package()
        self.package(self.root / "another-model", b"different detector")
        result = self.app.yolo_locations({})
        self.assertEqual(result["status"], "MULTIPLE")
        self.assertEqual(result["model_package"], "")

    def test_duplicate_identical_package_is_unambiguous(self):
        self.package()
        self.package(self.app.workspace / "models" / "copied-model")
        self.assertEqual(self.app.yolo_locations({})["status"], "FOUND")

    def test_incomplete_and_linked_packages_are_not_discovered(self):
        path, _ = self.package(self.root / "outside" / "hidden")
        (self.root / "linked-model").symlink_to(path, target_is_directory=True)
        s._write_json(self.root / "incomplete" / "MODEL_MANIFEST.json", {})
        self.assertEqual(self.app.yolo_locations({})["status"], "MISSING")

    def test_attestation_recovers_older_saved_model_location(self):
        path, sha = self.package(self.root / "outside" / "old-package")
        model_set = self.app.workspace / "rounds" / "round_old" / "model_set"
        attestation = model_set.parent / "attestation.json"
        s._write_json(attestation, {"model_package_root": str(path)})
        s._write_json(model_set / "MODEL_SET.json", {"yolo_attestation": str(attestation)})
        result = self.app.yolo_locations({"latest_round": {
            "model_set": str(model_set), "yolo_checkpoint_sha256": sha}})
        self.assertEqual(result["model_package"], str(path))

    def test_python_discovery_preserves_venv_symlink(self):
        child = self.app.workspace / "yolo_segment_env/bin/python"
        child.parent.mkdir(parents=True)
        child.symlink_to(Path(__import__('sys').executable))
        self.assertEqual(self.app.yolo_locations({})["python"], str(child))

    def attach_fixture(self):
        path, sha = self.package()
        previous = {"round_id": "round_xgb", "model_set": str(self.app.workspace / "original"),
                    "snapshot": "immutable-snapshot", "classifier_sha256": "unchanged-xgb"}
        settings = {"latest_round": previous}
        selected = {"snapshot": "immutable-snapshot", "xgb_bundle": "immutable-xgb",
                    "native_model": "immutable-native"}
        calls = []
        def run(command, label, **kwargs):
            calls.append(command)
            if any(str(arg).endswith("setup_yolo_segment_env.py") for arg in command):
                python = self.app.workspace / "yolo_segment_env/bin/python"
                python.parent.mkdir(parents=True, exist_ok=True)
                python.touch()
                return json.dumps({"status": "PASS", "python": str(python)})
            return "{}"
        def activated(folder):
            mode = calls[-1][calls[-1].index("--yolo-mode") + 1]
            return {"yolo_mode": mode, "yolo_status": "YOLO_EXTERNAL_ATTESTED" if mode != "off" else "YOLO_NOT_REQUESTED",
                    "yolo_checkpoint_sha256": sha if mode != "off" else None}
        return path, sha, settings, selected, calls, run, activated

    def test_blank_activation_remembers_paths_and_reuses_them_next_round(self):
        path, sha, settings, selected, calls, run, activated = self.attach_fixture()
        with patch.object(self.app, "_require_ready", return_value=settings), \
             patch.object(self.app, "_latest_model_set", side_effect=lambda x: (x["latest_round"], selected)), \
             patch.object(self.app, "_run", side_effect=run), \
             patch.object(self.app, "_model_set", side_effect=activated):
            first = self.app.attach_external_yolo(None, None, "both")
            self.assertEqual(first["yolo_model_package"], str(path))
            self.assertEqual(settings["yolo_defaults"]["checkpoint_sha256"], sha)
            self.assertEqual(first["classifier_sha256"], "unchanged-xgb")
            self.assertEqual(first["snapshot"], "immutable-snapshot")
            self.app.attach_external_yolo(None, None, "off")
            second = self.app.attach_external_yolo(None, None, "fusion")
            self.assertEqual(second["yolo_python"], first["yolo_python"])
            self.assertEqual(second["yolo_model_package"], str(path))
            setups = [cmd for cmd in calls if any(str(x).endswith("setup_yolo_segment_env.py") for x in cmd)]
            self.assertEqual(len(setups), 1)
            self.assertEqual(s._read_json(self.app.settings_path)["latest_round"]["yolo_mode"], "fusion")

    def test_discovered_weights_are_checksum_verified_before_execution(self):
        path, sha, settings, selected, calls, run, activated = self.attach_fixture()
        (path / "weights/best.pt").write_bytes(b"tampered")
        with patch.object(self.app, "_require_ready", return_value=settings), \
             patch.object(self.app, "_latest_model_set", return_value=(settings["latest_round"], selected)), \
             patch.object(self.app, "_run", side_effect=run):
            with self.assertRaisesRegex(s.StartError, "differs from its package manifest"):
                self.app.attach_external_yolo(None, None, "both")
        self.assertEqual(calls, [])
        self.assertNotIn("yolo_defaults", settings)

    def test_browser_accepts_blank_auto_location_without_running_inference(self):
        app = SimpleNamespace(attach_external_yolo=lambda *args: args)
        bridge = SimpleNamespace(backend=s, start_job=lambda action, fn: (action, fn(app)))
        self.assertEqual(web._Bridge.attach_yolo(bridge, {
            "model_package": "", "yolo_python": "", "mode": "both"}),
            ("attach_yolo", (None, None, "both")))

    @unittest.skipUnless(shutil.which("node"), "Node is needed for the browser event regression")
    def test_browser_autofill_blank_activation_and_stale_error(self):
        script = r'''
const vm = require('node:vm'), assert = require('node:assert/strict');
const input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const elements = {};
function el(id) {
  if (!elements[id]) elements[id] = {value:'', checked:false, options:[], files:[],
    listeners:{}, classList:{toggle(){}},
    addEventListener(event, callback){this.listeners[event] = callback;},
    replaceChildren(){this.options=[];}, append(option){this.options.push(option);},
    removeAttribute(){}, setAttribute(){}};
  return elements[id];
}
el('yoloMode').options = ['off','both','fusion','prompt'].map(value => ({value, textContent:value}));
const state = {ready:true, paths:{prefix:'/env', workspace:'/work'}, cards:['photo'], sessions:[],
  latest_round:{round_id:'round_a', model_set:'/round_a/model_set', yolo_mode:'off'},
  training_evidence:{status:'VERIFIED', yolo_mode:'off'}, review:{active:false},
  job:{id:1,status:'IDLE',action:'',logs:[]},
  yolo_locations:{status:'FOUND',model_package:'/models/saved-yolo',python:'/yolo/bin/python'}};
const requests = [];
const context = {document:{activeElement:null, getElementById:el,
  querySelectorAll:()=>[], querySelector:q=>q.startsWith('meta') ? {content:'csrf'} : {value:'example'},
  createElement:()=>({}), body:{classList:{toggle(){}}}},
  location:{pathname:'/t/test/'}, window:{scrollTo(){}}, setInterval(){},
  fetch:async (url, opts)=> {
    if (opts.method === 'POST') requests.push({url, body:JSON.parse(opts.body)});
    return {ok:true,json:async()=>opts.method === 'POST' ? {status:'STARTED'} : state};
  }};
vm.createContext(context);
vm.runInContext(input.js.replace('  refresh().then(() => {',
 '  globalThis.test = {render, run, announce};\n  refresh().then(() => {'), context);
(async()=>{
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(el('yoloPackage').value, '/models/saved-yolo');
  assert.equal(el('yoloPython').value, '/yolo/bin/python');
  el('yoloPackage').value = '';
  el('yoloPackage').listeners.input();
  el('yoloPython').value = '';
  el('yoloPython').listeners.input();
  el('yoloMode').value = 'both';
  await el('activateYolo').listeners.click();
  assert.equal(requests.at(-1).body.model_package, '');
  assert.equal(requests.at(-1).body.mode, 'both');
  assert.ok(requests.at(-1).url.endsWith('/attach-yolo'));
  const count = requests.length;
  await el('infer').listeners.click();
  assert.equal(requests.length, count); // Unapplied mode must not silently run off.
  assert.match(el('notice').textContent, /Apply your selected YOLO/);
  state.job = {id:2,status:'RUNNING',action:'infer',logs:[]};
  state.sessions = [{id:'session',status:'ANALYZING',yolo_mode:'off'}];
  context.test.render(state);
  assert.match(el('notice').textContent, /YOLO is off for this analysis/);
  assert.ok(el('activateYolo').disabled);
  context.test.announce('old error', true);
  await context.test.run(async()=>{});
  assert.notEqual(el('notice').textContent, 'old error');
  state.job = {id:3,status:'RUNNING',action:'train',logs:[]};
  context.test.render(state);
  assert.equal(el('stopJob').hidden, false);
  assert.equal(el('stopJob').disabled, false); // Stop remains usable during a busy job.
  assert.equal(el('train').disabled, true);
  await el('stopJob').listeners.click();
  assert.ok(requests.at(-1).url.endsWith('/stop-job'));
  state.job.status = 'STOPPING';
  context.test.render(state);
  assert.equal(el('stopJob').disabled, true);
  assert.equal(el('train').disabled, true);
  state.job.status = 'STOPPED';
  state.sessions = [{id:'saved',card:'photo',status:'SCORED'}];
  state.review_plan = {eligible_labeled_count:2,paper_ready:false,paper_reason:'Only 4 independent photos'};
  state.plan_session = 'saved';
  el('trainSelect').value = 'saved'; el('trainPolicy').value = 'paper';
  context.test.render(state);
  assert.equal(el('stopJob').hidden, true);
  assert.equal(el('train').disabled, true);
  assert.match(el('paperStatus').textContent, /Only 4 independent photos/);
  state.review_plan.paper_ready = true;
  context.test.render(state);
  assert.equal(el('train').disabled, false);
  el('exportSelect').value = 'saved'; el('exportScope').value = 'current';
  await el('exportCoco').listeners.click();
  assert.ok(requests.at(-1).url.endsWith('/export'));
  assert.equal(requests.at(-1).body.session_id, 'saved');
  assert.equal(requests.at(-1).body.scope, 'current');
  console.log('browser events PASS');
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
        result = subprocess.run([shutil.which("node"), "-e", script],
                                input=json.dumps({"js": web.JS}), text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_next_session_keeps_verified_yolo_and_saved_runtime(self):
        python = self.app.workspace / "yolo_segment_env/bin/python"
        python.parent.mkdir(parents=True)
        python.touch()
        card = self.root / "card"
        card.mkdir()
        latest = {"model_set": "unchanged-set", "snapshot": "unchanged-snapshot",
                  "yolo_python": str(python)}
        settings = {"latest_round": latest}
        with patch.object(self.app, "_require_ready", return_value=settings), \
             patch.object(self.app, "_latest_model_set", return_value=(latest, {"yolo_mode": "both"})), \
             patch.object(self.app, "resume_session", side_effect=self.app._session):
            session = self.app.create_session(card, use_latest=True)
        self.assertEqual(session["yolo_mode"], "both")
        self.assertEqual(session["yolo_python"], str(python))
        self.assertEqual(session["model_set"], "unchanged-set")

    def test_xgb_training_carries_yolo_locations_to_the_following_round(self):
        package, sha = self.package()
        python = self.app.workspace / "yolo_segment_env/bin/python"
        python.parent.mkdir(parents=True)
        python.touch()
        attestation = self.app.workspace / "attestation.json"
        s._write_json(attestation, {"model_package_root": str(package)})
        parent = {"yolo_mode": "both", "yolo_status": "YOLO_EXTERNAL_ATTESTED",
                  "yolo_checkpoint": str(package / "weights/best.pt"),
                  "yolo_checkpoint_sha256": sha, "yolo_attestation": str(attestation)}
        defaults = {"model_package": str(package), "python": str(python), "checkpoint_sha256": sha}
        settings = {"schema": "compag-easy-start-settings/v1", "version": s.VERSION,
                    "workspace": str(self.app.workspace), "yolo_defaults": defaults.copy()}
        session_dir = self.app.workspace / "sessions" / "card"
        review = session_dir / "review"
        review.mkdir(parents=True)
        session = {"path": str(session_dir), "status": "SCORED", "review_state": str(review),
                   "inference": "saved-inference", "parent_model_set": "parent-set",
                   "parent_snapshot": "parent-snapshot", "yolo_python": str(python)}
        off = {"yolo_mode": "off", "yolo_status": "YOLO_NOT_REQUESTED", "yolo_checkpoint_sha256": None}
        def run(command, label, **kwargs):
            if "plan-review" in command:
                return json.dumps({"eligible_labeled_count": 2, "bulk_accepted_count": 0})
            output = Path(command[command.index("--output") + 1])
            if "complete-xgb-round" in command:
                xgb = output / "xgb_model"
                xgb.mkdir(parents=True)
                (xgb / "classifier.ubj").write_bytes(b"synthetic fitted classifier")
                s._write_json(xgb / "PROJECT_MODEL_MANIFEST.json", {
                    "classifier_sha256": s._sha256(xgb / "classifier.ubj")})
                s._write_json(xgb / "TRAINING_RECIPE.json", {"fitted_boosting_rounds": 100})
                s._write_json(output / "training_snapshot.json", {"rows": [{}, {}]})
                s._write_json(output / "model_set/MODEL_SET.json", off)
                return json.dumps({"trained": {"row_count": 2, "boosting_rounds": 100}})
            s._write_json(output / "MODEL_SET.json", parent)
            return "{}"
        def model_set(folder):
            return parent if str(folder) == "parent-set" or Path(folder).name == "model_set_yolo" else off
        with patch.object(self.app, "_require_ready", return_value=settings), \
             patch.object(self.app, "_session", return_value=session), \
             patch.object(self.app, "_model_set", side_effect=model_set), \
             patch.object(self.app, "_run", side_effect=run):
            latest = self.app.train(session_dir)
        self.assertEqual(latest["yolo_mode"], "both")
        self.assertEqual(latest["yolo_model_package"], str(package))
        self.assertEqual(latest["yolo_python"], str(python))
        self.assertEqual(settings["yolo_defaults"], defaults)
        self.assertEqual(self.app.yolo_locations()["model_package"], str(package))


if __name__ == "__main__":
    unittest.main()
