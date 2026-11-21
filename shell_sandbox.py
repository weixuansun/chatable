"""Shell safety layer for the agent's run_shell tool.

Two independent defences:

1. A bubblewrap sandbox (when available): the command runs with read-only
   system directories, the workspace bind-mounted read-write at the same
   path, a private /tmp, home directories hidden, the chatable data dir
   hidden, a synthetic HOME, PID/IPC/UTS namespaces, and NO network by
   default (CHATTABLE_AGENT_SHELL_NET=1 opts back in). PATH directories —
   including a venv living outside the workspace — are re-mounted read-only
   so the configured toolchain keeps working.

2. An advisory deny-list (``denylist_reason``) that refuses obviously
   destructive commands outright. A deny-list is only a tripwire — the
   sandbox above is the real boundary.

CHATTABLE_AGENT_SANDBOX=auto|bwrap|off (auto = bwrap when available).
"""
from __future__ import annotations

import os
import re
import shutil
from typing import Optional

BWRAP = "bwrap"

# Env vars the sandboxed command never sees: credentials of the server, the
# model providers, and common cloud/CI secrets. (PATH/HOME/LANG/TERM are set
# explicitly via --setenv.)
_SCRUB_ENV_RE = re.compile(
    r"(?:_API_KEY|_SECRET|_TOKEN|_PASSWORD|_CREDENTIALS?|_AUTH)$"
    r"|^(?:AWS|GH|GITHUB|OPENAI|DEEPSEEK|ANTHROPIC)_"
    r"|^CHATABLE_AUTH_TOKEN$"
)


def scrubbed_env(env: Optional[dict] = None) -> dict:
    """os.environ minus credentials, for any agent shell (sandboxed or not)."""
    env = os.environ if env is None else env
    return {k: v for k, v in env.items() if not _SCRUB_ENV_RE.search(k)}


def bwrap_path() -> Optional[str]:
    return shutil.which(BWRAP)


def sandbox_mode() -> str:
    """Resolved sandbox mode: "bwrap" or "off" (auto → bwrap when present)."""
    raw = os.environ.get("CHATTABLE_AGENT_SANDBOX", "auto").strip().lower()
    if raw == "off":
        return "off"
    return "bwrap" if bwrap_path() else "off"


def sandbox_warning() -> str:
    """Non-empty when the user asked for a sandbox (auto/bwrap) but none ran."""
    raw = os.environ.get("CHATTABLE_AGENT_SANDBOX", "auto").strip().lower()
    if raw in ("auto", "bwrap") and not bwrap_path():
        return "bubblewrap (bwrap) not found — agent shell runs UNSANDBOXED"
    return ""


def shell_net_enabled() -> bool:
    return os.environ.get("CHATTABLE_AGENT_SHELL_NET", "").strip().lower() in ("1", "true", "yes")


def denylist_enabled() -> bool:
    return os.environ.get("CHATTABLE_AGENT_SHELL_DENYLIST", "").strip().lower() != "off"


# Advisory tripwires for obviously destructive commands. Each rule is a
# (regex, human reason) pair matched against the raw command string.
_DENY_RULES = [
    (re.compile(r"\brm\b[^\n|;&]*-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+(?:--\s+)?(?:/|~|\$HOME)(?:\s*$|[\s/])"),
     "rm -rf on / or the home directory"),
    (re.compile(r"\brm\b[^\n|;&]*-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\s+(?:--\s+)?(?:/|~|\$HOME)(?:\s*$|[\s/])"),
     "rm -fr on / or the home directory"),
    (re.compile(r"\bsudo\b"), "sudo is not available to the agent"),
    (re.compile(r"\b(?:curl|wget)\b[^\n]*\|\s*(?:sudo\s+)?(?:ba|z|da)?sh\b"),
     "piping a download straight into a shell"),
    (re.compile(r"\bchmod\b[^\n|;&]*-R[^\n|;&]*\s0?777\s+(?:/|~)", re.IGNORECASE),
     "recursive chmod 777 on / or home"),
    (re.compile(r"\bdd\b[^\n|;&]*\bof=/dev/"), "dd writing to a device"),
    (re.compile(r"\bmkfs(?:\.[a-z0-9]+)?\b"), "mkfs formats a filesystem"),
    (re.compile(r":\s*\(\s*\)\s*\{[^\n]*:\|:&"), "fork bomb"),
    (re.compile(r"\bgit\s+push\b[^\n|;&]*(?:--force|\s-f\b)"), "force-pushing to a remote"),
    (re.compile(r">>?\s*[^\n]*(?:\.ssh/|\.aws/|\.gnupg/)"),
     "writing into a credentials directory"),
]


def denylist_reason(command: str) -> Optional[str]:
    """A refusal reason when the command matches the deny-list, else None."""
    if not denylist_enabled():
        return None
    for rx, reason in _DENY_RULES:
        if rx.search(command):
            return f"command blocked by the deny-list ({reason}); this is advisory — the sandbox is the real boundary"
    return None


def build_bwrap_argv(command: str, ws_root: str, network: bool = False,
                     data_dir: str = "", env: Optional[dict] = None) -> list[str]:
    """The bwrap argv running ``command`` confined to *ws_root*.

    Read-only system dirs, private tmpfs /tmp, homes and the chatable data
    dir hidden, workspace rw at the same path, HOME redirected inside the
    sandbox, no network unless asked. PATH dirs (incl. a venv's parent when
    it looks like <venv>/bin) are re-mounted read-only after the home tmpfs
    so a toolchain living under /home keeps working.
    """
    env = os.environ if env is None else env
    argv: list[str] = [BWRAP, "--die-with-parent",
                       "--unshare-pid", "--unshare-ipc", "--unshare-uts"]
    if not network:
        argv.append("--unshare-net")
    # Read-only system. Symlinked dirs (/bin -> /usr/bin on merged-usr
    # systems) are skipped: bwrap cannot mount onto a symlink, and the
    # target is already bound.
    for d in ("/usr", "/bin", "/lib", "/lib64", "/etc", "/opt"):
        if os.path.exists(d) and not os.path.islink(d):
            argv += ["--ro-bind", d, d]
    # Merged-usr symlinked dirs: recreate the links (targets are bound above).
    for d in ("/bin", "/lib", "/lib64"):
        if os.path.islink(d):
            argv += ["--symlink", os.readlink(d), d]
    argv += ["--proc", "/proc", "--dev", "/dev"]
    # Private /tmp; homes hidden; data dir (settings, history) hidden.
    for d in ("/tmp", "/home", "/root"):
        if os.path.exists(d):
            argv += ["--tmpfs", d]
    if data_dir and os.path.exists(data_dir) and not (
            os.path.realpath(data_dir).startswith(("/tmp", "/home", "/root"))):
        argv += ["--tmpfs", data_dir]
    # Toolchain: every PATH dir back in, read-only (after the home tmpfs).
    # A venv/conda-style <root>/bin also gets its root (site-packages etc.).
    # Symlinked entries (/bin -> /usr/bin) are skipped: their targets are
    # already bound, and bwrap cannot mount onto a symlink.
    seen: set[str] = set()
    for entry in env.get("PATH", "").split(":"):
        entry = entry.rstrip("/")
        if not entry or entry == "/" or entry in seen \
                or not os.path.isdir(entry) or os.path.islink(entry):
            continue
        seen.add(entry)
        parent = os.path.dirname(entry)
        if os.path.basename(entry) == "bin" and parent and parent != "/" \
                and os.path.isdir(parent) and not os.path.islink(parent) \
                and parent not in seen:
            seen.add(parent)
            argv += ["--ro-bind", parent, parent]
        argv += ["--ro-bind", entry, entry]
    # The workspace itself: read-write, at its real path.
    argv += ["--bind", ws_root, ws_root]
    # A writable synthetic HOME inside the sandbox.
    argv += ["--dir", "/tmp/agent-home"]
    # Clean environment: only what the command legitimately needs.
    argv.append("--clearenv")
    argv += ["--setenv", "PATH", env.get("PATH", "/usr/bin:/bin")]
    argv += ["--setenv", "HOME", "/tmp/agent-home"]
    argv += ["--setenv", "TMPDIR", "/tmp"]
    argv += ["--setenv", "LANG", env.get("LANG", "C.UTF-8")]
    argv += ["--setenv", "TERM", env.get("TERM", "xterm")]
    argv += ["--chdir", ws_root, "--"]
    shell = "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"
    argv += [shell, "-c", command]
    return argv
