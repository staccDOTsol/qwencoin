"""A node's identity on the graph, as the node keeps it: the secret its keys come from and the
record of what it has published (run/nI/identity.json, mode 0600).

Two programs act for one identity: the node's agent (tools/chain.py, which bonds, attests and
signs a head per epoch) and the model when it drives the node (rails/). Both go through `use`,
which holds the node's lock, so the identity never signs from two diverging records and the
wallet is never spent from two places at once.
"""
import fcntl, json, os, sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import net  # noqa: E402


@contextmanager
def node_lock(i: int):
    """One holder at a time per node: whoever spends from its wallet or signs as its identity."""
    d = net.datadir(i)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "node.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _path(i: int) -> Path:
    return net.datadir(i) / "identity.json"


def read(i: int) -> dict | None:
    f = _path(i)
    return json.loads(f.read_text()) if f.exists() else None


class Identity:
    def __init__(self, node: int) -> None:
        self.node, self.agent = node, None

    @contextmanager
    def use(self):
        """(agent, state) under the node's lock; the state is written back on leaving."""
        from .agent import Agent

        with node_lock(self.node):
            st = read(self.node) or {"secret": os.urandom(32).hex(), "log": [], "seq": 0, "epoch": -1, "bond_sent": None}
            if self.agent is None:
                self.agent = Agent(self.node, "", secret=bytes.fromhex(st["secret"]))
            a = self.agent
            a.name = st.setdefault("name", "qwen-" + a.pid.hex()[:8])  # the keys come from the secret alone
            a.log, a.seq = st["log"], st["seq"]
            try:
                yield a, st
            finally:
                st.update(log=a.log, seq=a.seq)
                tmp = _path(self.node).with_suffix(".tmp")
                tmp.write_text(json.dumps(st))
                os.chmod(tmp, 0o600)
                os.replace(tmp, _path(self.node))
