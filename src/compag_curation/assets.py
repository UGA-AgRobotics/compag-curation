"""Explicit public asset registry, verification, and no-clobber acquisition."""

from __future__ import annotations

import hashlib
import os
import ssl
import stat
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

from .canonical.spec import EFFICIENT_GPU_PROFILE, FULL_IMAGE_GPU_PROFILE
from .public_io import PublicIOError, fsync_directory, hash_file_snapshot, rename_noreplace


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    kind: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    license_spdx: str
    license_url: str
    publisher: str
    source_commit: str
    bundled_in_repository: bool
    device_requirements: str
    acquisition_method: str


_ASSETS = {
    "sam2-apache-license": AssetSpec(
        asset_id="sam2-apache-license",
        kind="license_text",
        filename="SAM2-APACHE-2.0.txt",
        url="https://raw.githubusercontent.com/facebookresearch/sam2/2b90b9f5ceec907a1c18123530e92e794ad901a4/LICENSE",
        sha256="c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        size_bytes=11_357,
        license_spdx="Apache-2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        publisher="Meta Platforms, Inc.",
        source_commit="2b90b9f5ceec907a1c18123530e92e794ad901a4",
        bundled_in_repository=False,
        device_requirements="none",
        acquisition_method="explicit_https_fetch_only",
    ),
    "sam2.1-hiera-tiny-checkpoint": AssetSpec(
        asset_id="sam2.1-hiera-tiny-checkpoint",
        kind="model_checkpoint",
        filename="sam2.1_hiera_tiny.pt",
        url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        sha256="7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
        size_bytes=156_008_466,
        license_spdx="Apache-2.0",
        license_url="https://raw.githubusercontent.com/facebookresearch/sam2/2b90b9f5ceec907a1c18123530e92e794ad901a4/LICENSE",
        publisher="Meta Platforms, Inc.",
        source_commit="2b90b9f5ceec907a1c18123530e92e794ad901a4",
        bundled_in_repository=False,
        device_requirements="cuda_required_for_supported_v1.7_lite_profile",
        acquisition_method="explicit_https_fetch_only",
    ),
    "sam2.1-hiera-tiny-config": AssetSpec(
        asset_id="sam2.1-hiera-tiny-config",
        kind="model_configuration",
        filename="sam2.1_hiera_t.yaml",
        url="https://raw.githubusercontent.com/facebookresearch/sam2/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
        sha256="f932eac1c6241e910031b2f000a81cd9f8a8d4896e2277ab5ffb721f378b188d",
        size_bytes=3_855,
        license_spdx="Apache-2.0",
        license_url="https://raw.githubusercontent.com/facebookresearch/sam2/2b90b9f5ceec907a1c18123530e92e794ad901a4/LICENSE",
        publisher="Meta Platforms, Inc.",
        source_commit="2b90b9f5ceec907a1c18123530e92e794ad901a4",
        bundled_in_repository=False,
        device_requirements="none",
        acquisition_method="explicit_https_fetch_only",
    ),
    "sam2.1-hiera-large-checkpoint": AssetSpec(
        asset_id="sam2.1-hiera-large-checkpoint",
        kind="model_checkpoint",
        filename="sam2.1_hiera_large.pt",
        url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        sha256="2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
        size_bytes=898_083_611,
        license_spdx="Apache-2.0",
        license_url="https://raw.githubusercontent.com/facebookresearch/sam2/2b90b9f5ceec907a1c18123530e92e794ad901a4/LICENSE",
        publisher="Meta Platforms, Inc.",
        source_commit="2b90b9f5ceec907a1c18123530e92e794ad901a4",
        bundled_in_repository=False,
        device_requirements="cuda_required_for_supported_v1.7_full_profile",
        acquisition_method="explicit_https_fetch_only",
    ),
    "sam2.1-hiera-large-config": AssetSpec(
        asset_id="sam2.1-hiera-large-config",
        kind="model_configuration",
        filename="sam2.1_hiera_l.yaml",
        url="https://raw.githubusercontent.com/facebookresearch/sam2/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        sha256="1dbd6cb6dfebeaf588c7006ee222c6efbfa9049a7ad472a3cdfb2f5d919e8107",
        size_bytes=3_798,
        license_spdx="Apache-2.0",
        license_url="https://raw.githubusercontent.com/facebookresearch/sam2/2b90b9f5ceec907a1c18123530e92e794ad901a4/LICENSE",
        publisher="Meta Platforms, Inc.",
        source_commit="2b90b9f5ceec907a1c18123530e92e794ad901a4",
        bundled_in_repository=False,
        device_requirements="none",
        acquisition_method="explicit_https_fetch_only",
    ),
    "resnet50-imagenet1k-v2-weights": AssetSpec(
        asset_id="resnet50-imagenet1k-v2-weights",
        kind="model_checkpoint",
        filename="resnet50-11ad3fa6.pth",
        url="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        sha256="11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca",
        size_bytes=102_540_417,
        license_spdx="BSD-3-Clause",
        license_url="https://raw.githubusercontent.com/pytorch/vision/v0.28.0/LICENSE",
        publisher="PyTorch Foundation",
        source_commit="torchvision-v0.28.0",
        bundled_in_repository=False,
        device_requirements="cuda_required_for_supported_v1.7_lite_or_full",
        acquisition_method="explicit_https_fetch_only",
    ),
    "torchvision-bsd-license": AssetSpec(
        asset_id="torchvision-bsd-license",
        kind="license_text",
        filename="TORCHVISION-BSD-3-CLAUSE.txt",
        url="https://raw.githubusercontent.com/pytorch/vision/v0.28.0/LICENSE",
        sha256="6502f676851cfe25f8af75531dfb32375b7325b73c37e7b43741fa422893e71d",
        size_bytes=1_517,
        license_spdx="BSD-3-Clause",
        license_url="https://raw.githubusercontent.com/pytorch/vision/v0.28.0/LICENSE",
        publisher="PyTorch Foundation",
        source_commit="torchvision-v0.28.0",
        bundled_in_repository=False,
        device_requirements="none",
        acquisition_method="explicit_https_fetch_only",
    ),
}
_HOSTS = {"dl.fbaipublicfiles.com", "download.pytorch.org", "raw.githubusercontent.com"}

_CANONICAL_ASSETS = (
    "resnet50-imagenet1k-v2-weights",
    "sam2-apache-license",
    "sam2.1-hiera-large-checkpoint",
    "sam2.1-hiera-large-config",
    "torchvision-bsd-license",
)
_EFFICIENT_GPU_ASSETS = (
    "resnet50-imagenet1k-v2-weights",
    "sam2-apache-license",
    "sam2.1-hiera-tiny-checkpoint",
    "sam2.1-hiera-tiny-config",
    "torchvision-bsd-license",
)

_PROFILE_ASSETS = {
    "public-safe-balanced-v1": (
        "sam2-apache-license",
        "sam2.1-hiera-tiny-checkpoint",
        "sam2.1-hiera-tiny-config",
    ),
    "canonical-xgb-recall-cpu-v1": _CANONICAL_ASSETS,
    "canonical-xgb-recall-gpu-v1": _CANONICAL_ASSETS,
    EFFICIENT_GPU_PROFILE: _EFFICIENT_GPU_ASSETS,
    FULL_IMAGE_GPU_PROFILE: _CANONICAL_ASSETS,
}


def asset_registry() -> Mapping[str, AssetSpec]:
    return dict(_ASSETS)


def asset_ids_for_profile(profile: str) -> tuple[str, ...]:
    try:
        return _PROFILE_ASSETS[profile]
    except KeyError as exc:
        raise PublicIOError(f"unknown public profile: {profile}") from exc


def registry_payload() -> dict[str, object]:
    return {
        "schema": "compag-curation-public-asset-registry/v1",
        "automatic_download_on_import_or_validation": False,
        "assets": [asdict(_ASSETS[key]) for key in sorted(_ASSETS)],
    }


def _spec(asset_id: str) -> AssetSpec:
    try:
        return _ASSETS[asset_id]
    except KeyError as exc:
        raise PublicIOError(f"unknown public asset: {asset_id}") from exc


def verify_asset(root: Path, asset_id: str) -> dict[str, object]:
    spec = _spec(asset_id)
    absolute = root.absolute()
    info_root = absolute.lstat()
    if not stat.S_ISDIR(info_root.st_mode) or absolute.is_symlink() or absolute != absolute.resolve(strict=True):
        raise PublicIOError("asset root is unsafe or contains a symlink component")
    root = root.resolve(strict=True)
    path = root / spec.filename
    try:
        info = path.lstat()
    except OSError as exc:
        raise PublicIOError(f"asset is missing: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o644:
        raise PublicIOError(f"asset must be a single-link regular 0644 file: {path}")
    if info.st_size != spec.size_bytes:
        raise PublicIOError(f"asset size mismatch: {path}")
    digest, snapshot = hash_file_snapshot(path, require_single_link=True)
    if (
        snapshot.st_dev, snapshot.st_ino, snapshot.st_mode, snapshot.st_uid, snapshot.st_gid,
        snapshot.st_nlink, snapshot.st_size,
    ) != (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid, info.st_nlink, info.st_size):
        raise PublicIOError(f"asset changed before verification: {path}")
    if digest != spec.sha256:
        raise PublicIOError(f"asset SHA-256 mismatch: {path}")
    root_after = root.lstat()
    if (
        root_after.st_dev, root_after.st_ino, root_after.st_mode, root_after.st_uid, root_after.st_gid,
        root_after.st_mtime_ns, root_after.st_ctime_ns,
    ) != (
        info_root.st_dev, info_root.st_ino, info_root.st_mode, info_root.st_uid, info_root.st_gid,
        info_root.st_mtime_ns, info_root.st_ctime_ns,
    ):
        raise PublicIOError("asset root changed during verification")
    return {"status": "PASS", "asset_id": asset_id, "filename": spec.filename, "sha256": digest, "size_bytes": info.st_size}


def _validate_asset_url(url: str, role: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in _HOSTS
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 443}
    ):
        raise PublicIOError(f"{role} is outside the reviewed HTTPS host allowlist")


class _AllowlistedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        resolved = urllib.parse.urljoin(request.full_url, new_url)
        _validate_asset_url(resolved, "asset redirect")
        return super().redirect_request(request, file_pointer, code, message, headers, resolved)


def _default_open(url: str):
    _validate_asset_url(url, "asset URL")
    request = urllib.request.Request(url, headers={"User-Agent": "compag-curation/1.9.3"})
    opener = urllib.request.build_opener(
        _AllowlistedRedirectHandler(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    return opener.open(request, timeout=60)


def fetch_asset(
    root: Path,
    asset_id: str,
    *,
    opener: Callable[[str], object] | None = None,
) -> dict[str, object]:
    spec = _spec(asset_id)
    _validate_asset_url(spec.url, "asset URL")
    absolute = root.absolute()
    info_root = absolute.lstat()
    if not stat.S_ISDIR(info_root.st_mode) or absolute.is_symlink() or absolute != absolute.resolve(strict=True):
        raise PublicIOError("asset root is unsafe")
    root = absolute
    final = root / spec.filename
    if final.exists() or final.is_symlink():
        return verify_asset(root, asset_id)
    partial = root / f".{spec.filename}.{uuid.uuid4()}.partial"
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o644)
    digest = hashlib.sha256()
    total = 0
    selected_opener = opener or _default_open
    try:
        with selected_opener(spec.url) as response:
            final_url = getattr(response, "geturl", lambda: spec.url)()
            _validate_asset_url(final_url, "asset final URL")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > spec.size_bytes:
                    raise PublicIOError("asset exceeds its declared size")
                digest.update(chunk)
                offset = 0
                while offset < len(chunk):
                    written = os.write(fd, chunk[offset:])
                    if written <= 0:
                        raise PublicIOError("short write while acquiring asset")
                    offset += written
        if total != spec.size_bytes or digest.hexdigest() != spec.sha256:
            raise PublicIOError("downloaded asset identity mismatch")
        os.fchmod(fd, 0o644)
        os.fsync(fd)
        target = os.fstat(fd)
        if not stat.S_ISREG(target.st_mode) or target.st_nlink != 1 or stat.S_IMODE(target.st_mode) != 0o644:
            raise PublicIOError("downloaded asset staging file is unsafe")
    except (OSError, urllib.error.URLError) as exc:
        raise PublicIOError(f"asset acquisition failed: {asset_id}: {exc}") from exc
    finally:
        os.close(fd)
    partial_digest, partial_snapshot = hash_file_snapshot(partial, require_single_link=True)
    if partial_digest != spec.sha256 or partial_snapshot.st_size != spec.size_bytes:
        raise PublicIOError("downloaded asset failed destination verification")
    rename_noreplace(partial, final)
    fsync_directory(root)
    return verify_asset(root, asset_id)
