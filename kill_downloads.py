"""Kill stray corpus-download processes.

Why a script instead of a shell one-liner: a pattern-based kill in the shell
matches the shell's OWN command line (the pattern text lives in it), which took
out three of my own bash processes earlier. This file's command line contains no
matching text, and psutil excludes self and ancestors explicitly.
"""
import os
import psutil

me = psutil.Process(os.getpid())
protected = {me.pid} | {p.pid for p in me.parents()}

killed = []
for p in psutil.process_iter(["pid", "name", "cmdline"]):
    try:
        if p.pid in protected:
            continue
        name = (p.info["name"] or "").lower()
        cmd = " ".join(p.info["cmdline"] or [])
        if name == "cmd.exe" and "download_clean_corpora" in cmd:
            p.kill(); killed.append(("bat", p.pid))
        elif name.startswith("python") and "--local-dir" in cmd and "hf" in cmd.lower():
            p.kill(); killed.append(("hf", p.pid))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

for kind, pid in killed:
    print(f"killed {kind} pid {pid}")
print(f"{len(killed)} process(es) stopped")
