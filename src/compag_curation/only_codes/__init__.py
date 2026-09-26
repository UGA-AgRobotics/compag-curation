"""Only_codes drop-in compatibility workflow (profile ``only-codes-compat-v1``).

This package reproduces the computational behaviour of the supplied
Only_codes/Clean implementation (the ``COCO_2_features`` notebook + ``gate_core.py`` +
``sam2_and_filter.py`` + ``sam2_pipeline``) inside the application, accepting
the original project files directly.  It is an explicitly selected legacy
workflow; it does not change the canonical Full, Lite, Full-image or r92
workflows.  Legacy quirks are preserved deliberately and are documented in
:mod:`compag_curation.only_codes.contract` -- they are not scientific fixes.
"""

PROFILE_ID = "only-codes-compat-v1"
CONTRACT_SCHEMA = "compag-only-codes-compat-contract/v1"

__all__ = ["PROFILE_ID", "CONTRACT_SCHEMA"]
