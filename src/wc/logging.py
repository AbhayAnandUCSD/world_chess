"""W&B with automatic offline fallback (sync later with `wandb sync wandb/offline-*`)."""
import os

import wandb

PROJECT = os.environ.get("WANDB_PROJECT", "world-model-internal")


def _load_keyfile():
    # per-person key file (shared server accounts): ~/abhay/.wandb_key
    kf = os.path.expanduser("~/abhay/.wandb_key")
    if not os.environ.get("WANDB_API_KEY") and os.path.exists(kf):
        os.environ["WANDB_API_KEY"] = open(kf).read().strip()


def init(name: str, config: dict | None = None, group: str | None = None, job_type: str | None = None):
    _load_keyfile()
    # On shared servers set WC_KEYFILE_ONLY=1 so a ~/.netrc login belonging to someone else is never used.
    keyfile_only = os.environ.get("WC_KEYFILE_ONLY") == "1"
    online = bool(os.environ.get("WANDB_API_KEY")) or (not keyfile_only) and os.path.exists(os.path.expanduser("~/.netrc")) and \
        any(h in open(os.path.expanduser("~/.netrc")).read() for h in ("api.wandb.ai", "api.forge.coreweave.com"))
    kw = dict(project=PROJECT, name=name, config=config or {}, group=group, job_type=job_type,
              dir=os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        return wandb.init(mode=os.environ.get("WANDB_MODE", "online" if online else "offline"), **kw)
    except Exception as e:  # bad/missing key -> never kill the experiment, log offline
        print(f"[wandb] online init failed ({type(e).__name__}); falling back to offline", flush=True)
        os.environ.pop("WANDB_API_KEY", None)
        return wandb.init(mode="offline", **kw)
