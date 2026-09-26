#!/usr/bin/env python3
"""Preview the existing, unchanged paper trainer against saved review data."""
from pathlib import Path
import argparse
import importlib.util
import json
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--inference', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--parent-snapshot', type=Path)
    a = p.parse_args()
    from compag_curation.r92_project_model import finalize_review
    path = Path(__file__).with_name('train_paper_xgb_round.py')
    spec = importlib.util.spec_from_file_location('paper_fit_preflight', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        with tempfile.TemporaryDirectory(prefix='compag-paper-check-') as folder:
            root = Path(folder)
            snapshot = root / 'snapshot.json'
            finalize_review(a.state, a.inference, a.model, snapshot,
                            parent_snapshot=a.parent_snapshot, review_origin='human',
                            confirm_review_complete=True, scope='reviewed')
            report, _ = module.preflight(snapshot, a.model, root / 'unused-model-output')
        result = {'paper_ready': True,
                  'paper_reason': f"Ready: {report['train_card_count']} training photos, "
                  f"{report['heldout_card_count']} held-out photos, and five valid group folds.",
                  'preflight': report}
    except (ValueError, OSError, RuntimeError) as exc:
        result = {'paper_ready': False, 'paper_reason': str(exc)}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
