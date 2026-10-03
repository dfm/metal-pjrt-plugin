#!/usr/bin/env python3
# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0

"""The release workflow's test job: metal_pjrt_core and metal_pjrt_plugin
must import from the installed wheels, not from a source tree."""
import metal_pjrt_core
import metal_pjrt_plugin

for m in (metal_pjrt_core, metal_pjrt_plugin):
    assert "site-packages" in m.__file__, m.__file__
    print(m.__name__, m.__file__)
