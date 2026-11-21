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

A macOS equivalent uses Seatbelt (``/usr/bin/sandbox-exec``): see
``build_seatbelt_argv`` — writes confined to the workspace plus a per-run
temp dir / synthetic HOME, home directories and the data dir unreadable
(except the workspace and the toolchain's venv), and no network by default.

CHATTABLE_AGENT_SANDBOX=auto|bwrap|seatbelt|off (auto = bwrap on Linux,
seatbelt on macOS, when available).
CHATTABLE_AGENT_SANDBOX_PATH_HOME=1 also re-mounts non-venv PATH dirs
that live under $HOME (skipped by default: ~/.local/bin, tool config
dirs, etc. may hold credentials).
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from typing import Optional

BWRAP = "bwrap"
SANDBOX_EXEC = "/usr/bin/sandbox-exec"

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


def seatbelt_path() -> Optional[str]:
    """sandbox-exec on macOS (never on other platforms)."""
    if sys.platform == "darwin" and os.path.exists(SANDBOX_EXEC):
        return SANDBOX_EXEC
    return None


def _requested_sandbox() -> str:
    raw = os.environ.get("CHATTABLE_AGENT_SANDBOX", "auto").strip().lower()
    return raw if raw in ("auto", "bwrap", "seatbelt", "off") else "auto"


def sandbox_mode() -> str:
    """Resolved sandbox mode: "bwrap", "seatbelt" or "off".

    auto → bwrap on Linux when installed, seatbelt on macOS when
    /usr/bin/sandbox-exec exists, else off. An explicit backend that is not
    available resolves to "off" (and ``sandbox_warning`` says so).
    """
    raw = _requested_sandbox()
    if raw == "off":
        return "off"
    if raw in ("auto", "bwrap") and sys.platform.startswith("linux") and bwrap_path():
        return "bwrap"
    if raw in ("auto", "seatbelt") and seatbelt_path():
        return "seatbelt"
    return "off"


def sandbox_warning() -> str:
    """Non-empty when the user asked for a sandbox but none is in effect."""
    raw = _requested_sandbox()
    if raw == "off" or sandbox_mode() != "off":
        return ""
    if raw == "seatbelt" or (raw == "auto" and sys.platform == "darwin"):
        return "macOS sandbox-exec not found — agent shell runs UNSANDBOXED"
    if raw == "bwrap" or sys.platform.startswith("linux"):
        return "bubblewrap (bwrap) not found — agent shell runs UNSANDBOXED"
    return f"no shell sandbox available on {sys.platform} — agent shell runs UNSANDBOXED"


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


def _is_venv_root(path: str) -> bool:
    """A Python venv (pyvenv.cfg) or conda env (conda-meta/) root."""
    return os.path.isfile(os.path.join(path, "pyvenv.cfg")) or \
        os.path.isdir(os.path.join(path, "conda-meta"))


def _is_home_dir(path: str, env: dict) -> bool:
    """True for $HOME itself, /root, or a /home/<user> directory."""
    rp = os.path.realpath(path)
    home = os.path.realpath(env.get("HOME", "") or "x")
    if home != "x" and rp == home:
        return True
    if rp == "/root":
        return True
    return bool(re.match(r"^/home/[^/]+$", rp))


def build_bwrap_argv(command: str, ws_root: str, network: bool = False,
                     data_dir: str = "", env: Optional[dict] = None) -> list[str]:
    """The bwrap argv running ``command`` confined to *ws_root*.

    Read-only system dirs, private tmpfs /tmp, homes and the chatable data
    dir hidden, workspace rw at the same path, HOME redirected inside the
    sandbox, no network unless asked. PATH dirs are re-mounted read-only so
    the configured toolchain keeps working — narrowly: a parent dir only
    when it is a real venv/conda root (never a home dir or the data dir).
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
    # Toolchain: PATH dirs back in, read-only (after the home tmpfs) — but
    # narrowly, so home-local credential stores are never exposed: a parent
    # dir is bound only when it is clearly a Python venv (pyvenv.cfg) or
    # conda env (conda-meta/) and never a home dir or the data dir; PATH
    # entries under $HOME that are not venv bins are skipped entirely unless
    # CHATTABLE_AGENT_SANDBOX_PATH_HOME=1.
    home = os.path.realpath(env.get("HOME", "") or "") or None
    allow_home = os.environ.get("CHATTABLE_AGENT_SANDBOX_PATH_HOME", "") \
        .strip().lower() in ("1", "true", "yes")
    dd = os.path.realpath(data_dir) if data_dir else None
    seen: set[str] = set()
    for entry in env.get("PATH", "").split(":"):
        entry = entry.rstrip("/")
        if not entry or entry == "/" or entry in seen \
                or not os.path.isdir(entry) or os.path.islink(entry):
            continue
        real = os.path.realpath(entry)
        if dd and (real == dd or real.startswith(dd + os.sep)):
            continue  # never expose the data dir through PATH
        parent = os.path.dirname(entry)
        parent_is_env = (
            os.path.basename(entry) == "bin"
            and parent and _is_venv_root(parent)
            and not _is_home_dir(parent, env)
            and not (dd and os.path.realpath(parent) == dd)
        )
        if home and real.startswith(home + os.sep) and not parent_is_env \
                and not allow_home:
            continue  # home-local tools stay hidden by default
        seen.add(entry)
        if parent_is_env and parent not in seen:
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


# ---------------------------------------------------------------------------
# macOS: Seatbelt (sandbox-exec)
# ---------------------------------------------------------------------------
#
# Policy shape (mirrors the bwrap semantics above as closely as Seatbelt
# allows; same idea as the "(allow default) + targeted denies" profiles used
# by git-paw / Zed and the read/write split of Codex CLI's seatbelt policy):
#
#   (allow default)            processes, mach, sysctl, system reads keep
#                              working — a (deny default) profile is too
#                              brittle across macOS releases
#   (deny file-write*)         nothing is writable ...
#   (allow file-write* ...)    ... except the workspace, the per-run temp dir
#                              (TMPDIR + synthetic HOME) and the tty/null
#                              devices
#   (deny file-read-data ...)  /Users, /var/root, $HOME and the chatable data
#                              dir are unreadable (metadata stays readable so
#                              path resolution of a workspace under ~ works) ...
#   (allow file-read* ...)     ... except the workspace, the per-run temp dir
#                              and the toolchain (venv roots on PATH and the
#                              base Python they point at)
#   (deny network*)            no network at all (IP *and* unix sockets, so
#                              e.g. /var/run/docker.sock is unreachable) unless
#                              CHATTABLE_AGENT_SHELL_NET=1
#
# In SBPL later rules win, so each re-allow follows its deny. All paths are
# realpath'd (/var → /private/var, /tmp → /private/tmp) because Seatbelt
# matches resolved paths, and passed as -D parameters (no string splicing for
# the fixed paths) or as escaped string literals for variable-length lists.


def _sbpl_str(path: str) -> str:
    """An SBPL string literal (backslash and double quote escaped)."""
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _venv_base_home(venv_root: str) -> Optional[str]:
    """The base Python install a venv points at (``home =`` in pyvenv.cfg)."""
    try:
        with open(os.path.join(venv_root, "pyvenv.cfg"), encoding="utf-8") as fh:
            for line in fh:
                key, _, value = line.partition("=")
                if key.strip() == "home" and value.strip():
                    # home = <prefix>/bin → readable prefix
                    return os.path.dirname(os.path.realpath(value.strip()))
    except OSError:
        return None
    return None


def seatbelt_read_allow_paths(env: dict, data_dir: str = "") -> list[str]:
    """Toolchain dirs re-allowed for reading inside the hidden home dirs.

    Same narrowing as the bwrap PATH re-mounts: venv/conda roots on PATH
    (plus the base interpreter a venv points at); other PATH dirs under
    $HOME only with CHATTABLE_AGENT_SANDBOX_PATH_HOME=1. Never the data dir.
    """
    home = os.path.realpath(env.get("HOME", "") or "") or None
    allow_home = os.environ.get("CHATTABLE_AGENT_SANDBOX_PATH_HOME", "") \
        .strip().lower() in ("1", "true", "yes")
    dd = os.path.realpath(data_dir) if data_dir else None
    out: list[str] = []

    def add(p: Optional[str]) -> None:
        if not p:
            return
        rp = os.path.realpath(p)
        if rp in ("/", home) or rp in out:
            return
        if dd and (rp == dd or rp.startswith(dd + os.sep) or dd.startswith(rp + os.sep)):
            return
        out.append(rp)

    for entry in env.get("PATH", "").split(":"):
        entry = entry.rstrip("/")
        if not entry or entry == "/" or not os.path.isdir(entry):
            continue
        parent = os.path.dirname(entry)
        if os.path.basename(entry) == "bin" and parent and _is_venv_root(parent) \
                and not _is_home_dir(parent, env):
            add(parent)
            add(_venv_base_home(parent))
        elif allow_home:
            add(entry)
    return out


def build_seatbelt_profile(network: bool = False, read_allow: Optional[list[str]] = None,
                           hidden: Optional[list[str]] = None) -> str:
    """The SBPL profile text. Fixed paths come in as -D params:
    WORKSPACE, RUN_TMP (per-run TMPDIR + synthetic HOME). *hidden* are
    unreadable roots (homes, data dir); *read_allow* re-allows toolchain
    dirs inside them."""
    lines = [
        "(version 1)",
        "; chatable agent shell sandbox (macOS Seatbelt)",
        "(allow default)",
        "",
        "; --- writes: workspace + per-run temp/home + terminal devices only",
        "(deny file-write*)",
        "(allow file-write*",
        '  (subpath (param "WORKSPACE"))',
        '  (subpath (param "RUN_TMP"))',
        '  (literal "/dev/null") (literal "/dev/zero")',
        '  (literal "/dev/tty") (regex #"^/dev/ttys[0-9]+$")',
        '  (literal "/dev/dtracehelper") (regex #"^/dev/fd/"))',
    ]
    if hidden:
        lines += ["", "; --- reads: homes and the chatable data dir are hidden ...",
                  "(deny file-read-data file-read-xattr"]
        lines += [f"  (subpath {_sbpl_str(p)})" for p in hidden]
        lines[-1] += ")"
        lines += ["; ... except the workspace, the run temp dir and the toolchain",
                  "(allow file-read*",
                  '  (subpath (param "WORKSPACE"))',
                  '  (subpath (param "RUN_TMP"))']
        lines += [f"  (subpath {_sbpl_str(p)})" for p in (read_allow or [])]
        lines[-1] += ")"
    if not network:
        lines += ["", "; --- network: none (IP and unix-domain sockets)",
                  "(deny network*)"]
    return "\n".join(lines) + "\n"


def build_seatbelt_argv(command: str, ws_root: str, run_tmp: str, network: bool = False,
                        data_dir: str = "", env: Optional[dict] = None
                        ) -> tuple[list[str], dict]:
    """(argv, env) running ``command`` under sandbox-exec confined to *ws_root*.

    *run_tmp* is a per-run directory created (and removed) by the caller;
    the command gets ``<run_tmp>/tmp`` as TMPDIR and ``<run_tmp>/home`` as
    HOME. The returned env is a clean one (PATH/HOME/TMPDIR/LANG/TERM), like
    bwrap's --clearenv.
    """
    env = os.environ if env is None else env
    ws = os.path.realpath(ws_root)
    rt = os.path.realpath(run_tmp)
    hidden: list[str] = []
    for d in ("/Users", "/var/root", env.get("HOME", "")):
        if d:
            rd = os.path.realpath(d)
            if rd not in hidden and rd != "/":
                hidden.append(rd)
    if data_dir:
        rd = os.path.realpath(data_dir)
        # Hidden even when the workspace (an orchestrator run worktree) lives
        # inside it: the WORKSPACE re-allow below comes later and wins.
        if rd not in hidden and rd != "/":
            hidden.append(rd)
    profile = build_seatbelt_profile(
        network=network,
        read_allow=seatbelt_read_allow_paths(env, data_dir),
        hidden=hidden,
    )
    shell = "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"
    argv = [SANDBOX_EXEC, "-p", profile,
            "-D", f"WORKSPACE={ws}", "-D", f"RUN_TMP={rt}",
            shell, "-c", command]
    clean = {
        "PATH": env.get("PATH", "/usr/bin:/bin"),
        "HOME": os.path.join(rt, "home"),
        "TMPDIR": os.path.join(rt, "tmp"),
        "LANG": env.get("LANG", "en_US.UTF-8"),
        "TERM": env.get("TERM", "xterm"),
    }
    return argv, clean


def make_run_tmp() -> str:
    """A fresh per-run temp dir with home/ and tmp/ inside (caller removes it)."""
    d = tempfile.mkdtemp(prefix="chatable-sb-")
    os.makedirs(os.path.join(d, "home"), exist_ok=True)
    os.makedirs(os.path.join(d, "tmp"), exist_ok=True)
    return d
