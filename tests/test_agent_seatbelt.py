"""macOS Seatbelt backend: mode resolution, profile text and argv (mocked —
CI is Linux, sandbox-exec cannot run here)."""
import os
import subprocess

import pytest

import shell_sandbox as sb
from agent_tools import do_run_shell
from workspace import Workspace


def _darwin(monkeypatch, has_exec=True, has_bwrap=False):
    monkeypatch.setattr(sb.sys, "platform", "darwin")
    real_exists = os.path.exists
    monkeypatch.setattr(sb.os.path, "exists",
                        lambda p: has_exec if p == sb.SANDBOX_EXEC else real_exists(p))
    monkeypatch.setattr(sb.shutil, "which", lambda name: "/usr/bin/bwrap" if has_bwrap else None)


def test_mode_auto_darwin_seatbelt(monkeypatch):
    _darwin(monkeypatch)
    monkeypatch.setenv("CHATTABLE_AGENT_SANDBOX", "auto")
    assert sb.sandbox_mode() == "seatbelt"
    assert sb.sandbox_warning() == ""


def test_mode_explicit_seatbelt_and_off(monkeypatch):
    _darwin(monkeypatch)
    monkeypatch.setenv("CHATTABLE_AGENT_SANDBOX", "seatbelt")
    assert sb.sandbox_mode() == "seatbelt"
    monkeypatch.setenv("CHATTABLE_AGENT_SANDBOX", "off")
    assert sb.sandbox_mode() == "off"
    assert sb.sandbox_warning() == ""


def test_mode_darwin_without_sandbox_exec_warns(monkeypatch):
    _darwin(monkeypatch, has_exec=False)
    monkeypatch.setenv("CHATTABLE_AGENT_SANDBOX", "auto")
    assert sb.sandbox_mode() == "off"
    assert "sandbox-exec" in sb.sandbox_warning()


def test_bwrap_never_used_on_darwin(monkeypatch):
    _darwin(monkeypatch, has_exec=False, has_bwrap=True)
    monkeypatch.setenv("CHATTABLE_AGENT_SANDBOX", "bwrap")
    assert sb.sandbox_mode() == "off"
    assert sb.sandbox_warning()


def test_seatbelt_not_used_on_linux(monkeypatch):
    monkeypatch.setattr(sb.sys, "platform", "linux")
    monkeypatch.setattr(sb.shutil, "which", lambda name: None)
    monkeypatch.setenv("CHATTABLE_AGENT_SANDBOX", "seatbelt")
    assert sb.sandbox_mode() == "off"
    assert "sandbox-exec" in sb.sandbox_warning()
    monkeypatch.setenv("CHATTABLE_AGENT_SANDBOX", "auto")
    assert "bwrap" in sb.sandbox_warning()


def test_mode_linux_bwrap(monkeypatch):
    monkeypatch.setattr(sb.sys, "platform", "linux")
    monkeypatch.setattr(sb.shutil, "which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setenv("CHATTABLE_AGENT_SANDBOX", "auto")
    assert sb.sandbox_mode() == "bwrap"


def test_profile_structure_no_net():
    prof = sb.build_seatbelt_profile(network=False, read_allow=["/opt/py"], hidden=["/Users"])
    assert prof.startswith("(version 1)")
    assert "(allow default)" in prof
    # write deny precedes the narrow re-allow (later rules win in SBPL)
    assert prof.index("(deny file-write*)") < prof.index("(allow file-write*")
    assert '(subpath (param "WORKSPACE"))' in prof and '(subpath (param "RUN_TMP"))' in prof
    assert '(literal "/dev/null")' in prof
    assert prof.index("(deny file-read-data") < prof.index("(allow file-read*")
    assert '(subpath "/opt/py")' in prof
    assert "(deny network*)" in prof
    assert prof.count("(") == prof.count(")")


def test_profile_network_on_has_no_network_rule():
    prof = sb.build_seatbelt_profile(network=True, hidden=["/Users"])
    assert "network" not in prof.replace("; --- network", "")


def test_sbpl_string_escaping():
    assert sb._sbpl_str('/a "b"\\c') == '"/a \\"b\\"\\\\c"'
    prof = sb.build_seatbelt_profile(hidden=['/Users/x "y'])
    assert '(subpath "/Users/x \\"y")' in prof


def test_argv_and_clean_env(tmp_path, monkeypatch):
    ws = tmp_path / "ws"; ws.mkdir()
    run_tmp = tmp_path / "rt"; run_tmp.mkdir()
    home = tmp_path / "home"; home.mkdir()
    data = tmp_path / "data"; data.mkdir()
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "DEEPSEEK_API_KEY": "sk-x",
           "LANG": "en_US.UTF-8"}
    argv, cenv = sb.build_seatbelt_argv("echo hi", str(ws), str(run_tmp),
                                        network=False, data_dir=str(data), env=env)
    assert argv[0] == "/usr/bin/sandbox-exec" and argv[1] == "-p"
    prof = argv[2]
    assert f"WORKSPACE={os.path.realpath(ws)}" in argv
    assert f"RUN_TMP={os.path.realpath(run_tmp)}" in argv
    assert argv[-2:] == ["-c", "echo hi"]
    assert f'(subpath "{os.path.realpath(home)}")' in prof      # $HOME hidden
    assert f'(subpath "{os.path.realpath(data)}")' in prof      # data dir hidden
    assert '(subpath "/Users")' in prof
    assert "(deny network*)" in prof
    assert set(cenv) == {"PATH", "HOME", "TMPDIR", "LANG", "TERM"}
    assert cenv["HOME"].startswith(os.path.realpath(run_tmp))
    assert cenv["TMPDIR"].startswith(os.path.realpath(run_tmp))
    assert "sk-x" not in " ".join(argv) + " ".join(cenv.values())


def test_realpath_resolution(tmp_path):
    real = tmp_path / "real"; real.mkdir()
    link = tmp_path / "link"; link.symlink_to(real)
    rt = tmp_path / "rt"; rt.mkdir()
    argv, _ = sb.build_seatbelt_argv("true", str(link), str(rt), env={"PATH": "/usr/bin"})
    assert f"WORKSPACE={os.path.realpath(real)}" in argv


def test_venv_toolchain_readable(tmp_path, monkeypatch):
    home = tmp_path / "home"
    base = home / ".pyenv" / "versions" / "3.12"
    (base / "bin").mkdir(parents=True)
    venv = home / "proj" / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(f"home = {base / 'bin'}\nversion = 3.12\n")
    other = home / ".local" / "bin"; other.mkdir(parents=True)
    env = {"HOME": str(home), "PATH": f"{venv/'bin'}:{other}:/usr/bin"}
    monkeypatch.delenv("CHATTABLE_AGENT_SANDBOX_PATH_HOME", raising=False)
    allow = sb.seatbelt_read_allow_paths(env)
    assert os.path.realpath(venv) in allow
    assert os.path.realpath(base) in allow          # base interpreter of the venv
    assert os.path.realpath(other) not in allow     # home-local tools stay hidden
    monkeypatch.setenv("CHATTABLE_AGENT_SANDBOX_PATH_HOME", "1")
    assert os.path.realpath(other) in sb.seatbelt_read_allow_paths(env)


def test_data_dir_never_readallowed(tmp_path):
    data = tmp_path / "data"
    venv = data / "venv"; (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("version = 3.12\n")
    env = {"HOME": str(tmp_path / "h"), "PATH": f"{venv/'bin'}:/usr/bin"}
    assert sb.seatbelt_read_allow_paths(env, data_dir=str(data)) == []


def test_run_shell_uses_seatbelt_and_cleans_tmp(tmp_path, monkeypatch):
    """do_run_shell builds the sandbox-exec argv + clean env and removes the
    per-run temp dir afterwards (Popen is faked: no sandbox-exec on Linux)."""
    import agent_tools
    monkeypatch.setattr(agent_tools, "sandbox_mode", lambda: "seatbelt")
    seen = {}
    real_popen = subprocess.Popen

    def fake_popen(args, **kw):
        seen["args"], seen["env"] = args, kw["env"]
        return real_popen(["/bin/sh", "-c", "echo sandboxed"], **{**kw, "env": kw["env"]})

    monkeypatch.setattr(agent_tools.subprocess, "Popen", fake_popen)
    ws = Workspace(str(tmp_path / "ws"))
    r = do_run_shell(ws, "echo hi")
    assert r.ok and "sandboxed" in r.output
    assert seen["args"][0] == sb.SANDBOX_EXEC
    rt = next(a.split("=", 1)[1] for a in seen["args"] if a.startswith("RUN_TMP="))
    assert not os.path.exists(rt)                    # cleaned up
    assert set(seen["env"]) == {"PATH", "HOME", "TMPDIR", "LANG", "TERM"}


def test_run_shell_seatbelt_denylist_still_applies(tmp_path, monkeypatch):
    import agent_tools
    monkeypatch.setattr(agent_tools, "sandbox_mode", lambda: "seatbelt")
    ws = Workspace(str(tmp_path / "ws"))
    r = do_run_shell(ws, "sudo rm -rf /")
    assert not r.ok and "deny-list" in r.output
