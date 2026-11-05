import yaml
from .model import ROOT

import os


def load(name=None):
    name = name or os.environ.get("WC_CONFIG", "p1")
    c = yaml.safe_load(open(ROOT / "configs" / f"{name}.yaml"))
    c.setdefault("players", ["karpov", "kasparov"])
    return c
