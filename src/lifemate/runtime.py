"""Runtime helpers shared by the scripts: locating the split, importing zaic_tools,
safe incremental submission writing, model construction."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

from .asr import BaseASR, build_asr
from .config import Config
from .llm import BaseLLM, build_llm, set_all_seeds


# --------------------------------------------------------------------------- #
# Split discovery
# --------------------------------------------------------------------------- #

def _is_split_dir(p: Path) -> Optional[str]:
    """If p is a split folder (contains <name>.jsonl + <name>_tools_catalog.json), return <name>."""
    if not p.is_dir():
        return None
    for jl in p.glob("*.jsonl"):
        name = jl.stem
        if name == "sample_submission":
            continue
        if (p / f"{name}_tools_catalog.json").exists():
            return name
    return None


def locate_split(data_dir: str | Path, split: Optional[str] = None) -> tuple[Path, str, Path]:
    """Return (root, split_name, split_dir) such that ToolEnv(split_name, id, root=root) works.

    Handles both layouts:
      <data_dir>/<split>/...         (root = data_dir)
      <data_dir> IS the split folder  (organisers mount private_test at /data)
    In the second case a symlink <tmp>/<split_name> -> <data_dir> is created so that
    zaic_tools, which resolves root/<split>/<split>_environment, finds everything.
    """
    data_dir = Path(data_dir).resolve()
    if split and (data_dir / split).is_dir():
        return data_dir, split, data_dir / split
    name = _is_split_dir(data_dir)
    if name is not None:
        if data_dir.name == name:
            return data_dir.parent, name, data_dir
        tmp_root = Path(tempfile.mkdtemp(prefix="lifemate_root_"))
        link = tmp_root / name
        os.symlink(data_dir, link, target_is_directory=True)
        return tmp_root, name, link
    for child in sorted(data_dir.iterdir()):
        if child.is_dir() and _is_split_dir(child) == child.name:
            return data_dir, child.name, child
    raise FileNotFoundError(f"no split folder found under {data_dir}")


def import_toolenv(split_dir: Path):
    """Import the organisers' ToolEnv from THIS split's zaic_tools (each split ships its own)."""
    p = str(split_dir)
    if p not in sys.path:
        sys.path.insert(0, p)
    for mod in [m for m in sys.modules if m == "zaic_tools" or m.startswith("zaic_tools.")]:
        del sys.modules[mod]
    from zaic_tools import ToolEnv  # type: ignore
    return ToolEnv


# --------------------------------------------------------------------------- #
# Submission writer
# --------------------------------------------------------------------------- #

class SubmissionWriter:
    """Append-only JSONL writer: flush + fsync after every line so a crash or a kill
    still leaves every finished sample on disk (unfinished ones count as abstain/0)."""

    def __init__(self, path: str | Path, resume: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done: set[str] = set()
        if resume and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    self.done.add(json.loads(line)["id"])
                except Exception:
                    pass
        self._f = open(self.path, "a", encoding="utf-8")

    def write(self, sample_id: str, tool_calls: list, state_hash: str) -> None:
        line = json.dumps({"id": sample_id, "tool_calls": tool_calls, "state_hash": state_hash}, ensure_ascii=False)
        self._f.write(line + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())
        self.done.add(sample_id)

    def close(self) -> None:
        self._f.close()


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def build_components(cfg: Config, oracle_only: bool = False) -> tuple[BaseLLM, BaseASR]:
    set_all_seeds(cfg.seed)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    llm = build_llm(cfg)
    if oracle_only:
        from .asr import FakeASR
        asr: BaseASR = FakeASR()
    else:
        asr = build_asr(cfg)
    return llm, asr
