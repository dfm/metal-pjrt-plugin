#!/usr/bin/env python3
# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0

"""The release workflow's plan (.github/workflows/release.yml): checks the
versions, the tag and the core pin, and decides whether metal-pjrt-core is
built. Prints the job outputs (key=value) on stdout and a summary on stderr.

  REF=refs/tags/v0.1.0 scripts/release_plan.py  # needs Python 3.11+
"""
import os, re, subprocess, sys, tomllib, urllib.error, urllib.request

# The library's inputs: a change here needs a new metal-pjrt-core.
CORE_SOURCES = ["metal_pjrt", "core", "third_party", "MODULE.bazel",
                "MODULE.bazel.lock", ".bazelrc", ".bazelversion"]

def on_pypi(name, version):
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    try:
        urllib.request.urlopen(url, timeout=30)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise

def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)

def fail(msg):
    print(f"::error::{msg}", file=sys.stderr)
    sys.exit(1)

plugin = tomllib.load(open("pyproject.toml", "rb"))["project"]
core = tomllib.load(open("core/pyproject.toml", "rb"))["project"]
pv, cv = plugin["version"], core["version"]
pins = [d for d in plugin["dependencies"]
        if re.match(r"metal-pjrt-core\b", d)]
if not pins or not pins[0].startswith(f"metal-pjrt-core=={cv};"):
    fail(f"pyproject.toml must pin metal-pjrt-core=={cv}: {pins}")

ref = os.environ["REF"]
tagged = ref.startswith("refs/tags/")
if tagged:
    if ref != f"refs/tags/v{pv}":
        fail(f"tag {ref} does not match metal-pjrt-plugin {pv}")
    if git("merge-base", "--is-ancestor", "HEAD", "origin/main").returncode:
        fail(f"{ref} is not on main: tag a commit of main")
if on_pypi("metal-pjrt-plugin", pv):
    msg = f"metal-pjrt-plugin {pv} is already on PyPI: bump the version"
    if tagged:
        fail(msg)
    print(f"::notice::{msg} (dry run continues)", file=sys.stderr)

build_core = not on_pypi("metal-pjrt-core", cv)
if not build_core:
    # The published core must still be what the sources build: find
    # the first release tag with this core version.
    tags = git("tag", "--list", "v*", "--sort=creatordate").stdout.split()
    released = next(
        (t for t in tags
         if re.search(rf'^version = "{re.escape(cv)}"$',
                      git("show", f"{t}:core/pyproject.toml").stdout, re.M)),
        None)
    if released is None:
        fail(f"metal-pjrt-core {cv} is on PyPI but no v* tag has it")
    if git("diff", "--quiet", released, "HEAD", "--", *CORE_SOURCES).returncode:
        fail(f"the library's sources changed since metal-pjrt-core {cv} "
             f"was released ({released}): bump core/pyproject.toml "
             f"and the frontend's pin")
print(f"plugin_version={pv}")
print(f"core_version={cv}")
print(f"build_core={str(build_core).lower()}")
print(f"metal-pjrt-plugin {pv}; metal-pjrt-core {cv} "
      f"({'new: build it' if build_core else 'on PyPI: reuse it'})",
      file=sys.stderr)
