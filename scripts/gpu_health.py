#!/usr/bin/env python3
"""Report GPU watchdog resets recorded by the Metal runtime and quarantined
kernels; refuse (exit 1) with --strict when the GPU was reset since boot.

The runtime appends one JSON line per reset to ~/.cache/openmetal/gpu_resets.jsonl
(METAL_PJRT_STATE_DIR overrides the directory; a missing log starts as a copy
of the pre-rename ~/.cache/jax_metal/gpu_resets.jsonl) with the time, the plugin build
(LC_UUID of the dylib; diagnostics only) and the kernels that were in the
command buffer that timed out (built-in fill/copy kernels are listed but never
blamed). Kernels seen in two or more resets since boot are quarantined: the
runtime refuses to load them until a reboot or --clear. A kernel whose source
changed gets a new key; after a fix outside the kernel source (runtime, launch
dimensions), --clear. After a few resets the driver leaves the GPU slow until a reboot, so
benchmark timings are meaningless; bench/run_all.sh runs this with --strict.

  scripts/gpu_health.py            # summary
  scripts/gpu_health.py --strict   # exit 1 if any reset since boot
  scripts/gpu_health.py --clear    # forget resets and lift quarantines
"""
import collections, json, os, pathlib, shutil, subprocess, sys, time

STATE_DIR = pathlib.Path(os.environ.get("METAL_PJRT_STATE_DIR") or
                         pathlib.Path.home() / ".cache" / "openmetal")
LOG = STATE_DIR / "gpu_resets.jsonl"
# The same one-time migration as the runtime (Device::LoadResetLog).
OLD_LOG = pathlib.Path.home() / ".cache" / "jax_metal" / "gpu_resets.jsonl"
STRIKES = int(os.environ.get("METAL_PJRT_QUARANTINE_STRIKES", "2"))


def boot_time():
    out = subprocess.run(["sysctl", "-n", "kern.boottime"], capture_output=True, text=True).stdout
    # "{ sec = 1758800000, usec = 0 } ..."
    for tok in out.replace(",", " ").split():
        if tok.isdigit():
            return int(tok)
    return 0


def migrate():
    if not os.environ.get("METAL_PJRT_STATE_DIR") and not LOG.exists() and OLD_LOG.exists():
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OLD_LOG, LOG)


def load():
    if not LOG.exists():
        return []
    events = []
    for line in LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return events


def main(argv):
    migrate()
    if "--clear" in argv:
        # Truncate rather than delete, so that the migration does not bring
        # the old log back.
        if LOG.exists():
            LOG.write_text("")
        print(f"cleared {LOG}")
        return 0
    events = load()
    boot = boot_time()
    since_boot = [e for e in events if e.get("time", 0) >= boot]
    print(f"GPU resets recorded: {len(events)} total, {len(since_boot)} since boot "
          f"({time.strftime('%Y-%m-%d %H:%M', time.localtime(boot))})")
    for e in since_boot[-5:]:
        names = ", ".join(k.get("name", "?") for k in e.get("kernels", [])[:8])
        more = len(e.get("kernels", [])) - 8
        print(f"  {time.strftime('%m-%d %H:%M:%S', time.localtime(e['time']))} pid {e.get('pid')} "
              f"build {e.get('build', '?')[:8]}: "
              f"{names}{f' (+{more} more)' if more > 0 else ''}")
    strikes = collections.Counter(k["key"] for e in since_boot for k in e.get("kernels", []))
    names = {k["key"]: k.get("name", "?") for e in since_boot for k in e.get("kernels", [])}
    quarantined = [k for k, n in strikes.items() if STRIKES > 0 and n >= STRIKES]
    if quarantined:
        print(f"quarantined kernels ({len(quarantined)}; seen in >= {STRIKES} resets since "
              f"boot; a reboot lifts it; clear with: scripts/gpu_health.py --clear):")
        for k in quarantined[:20]:
            print(f"  {names[k]}  [{k}]")
    if since_boot:
        print("The GPU has been reset since boot; the driver leaves it slow after resets. "
              "Reboot before taking timings.")
        if "--strict" in argv:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
