# Security

Please report a suspected vulnerability privately, through GitHub's
"Report a vulnerability" on the repository's Security tab, rather than in a
public issue.

In scope: the plugin library and the `metal_pjrt_plugin` Python package,
for example memory corruption reachable from a JAX program, or from a
serialized executable or an entry of JAX's persistent compilation cache.
Bugs in JAX, XLA or Metal itself belong to those projects.

A GPU reset or a wrong numerical result is a bug, not a vulnerability:
please open an ordinary issue (`CONTRIBUTING.md`).
