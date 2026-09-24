"""Report what corpus-download processes are actually running (full command lines)."""
import psutil

me = psutil.Process()
protected = {me.pid} | {p.pid for p in me.parents()}
rows = []
for p in psutil.process_iter(["pid", "name", "cmdline"]):
    try:
        if p.pid in protected:
            continue
        name = (p.info["name"] or "").lower()
        cmd = " ".join(p.info["cmdline"] or [])
        if name == "cmd.exe" and "download_clean_corpora" in cmd:
            rows.append(("BAT", p.pid, cmd))
        elif name.startswith("python") and ("huggingface" in cmd.lower() or "hf" in cmd.lower()) \
                and ("download" in cmd.lower() or "--local-dir" in cmd):
            rows.append(("HF", p.pid, cmd))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

if not rows:
    print("no download processes running")
for kind, pid, cmd in rows:
    print(f"{kind} {pid}: {cmd[:300]}")
