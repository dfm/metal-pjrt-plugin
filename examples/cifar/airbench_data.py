# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0
# Derived from airbench94 (https://github.com/KellerJordan/cifar10-airbench):
# Copyright (c) 2024 Keller Jordan. MIT License; the full notice is in
# LICENSE-airbench in this directory.
"""CIFAR-10 loading and airbench94's hyperparameters, shared by airbench.py
(JAX) and torch_baseline.py (PyTorch); numpy only.

CIFAR-10 (Krizhevsky, "Learning Multiple Layers of Features from Tiny
Images", 2009) is downloaded once from the authors' site into
~/.cache/metal-pjrt-examples; it is not part of this repository.
"""
import hashlib, os, pickle, sys, tarfile, urllib.request

import numpy as np

DATA = os.path.expanduser("~/.cache/metal-pjrt-examples")
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
# SHA-256 of the archive (its MD5, c58f30108f718f92721af3b95e74349a, is the
# one torchvision checks) and of each batch file in it. The batch files are
# pickles, so each is checked before it is unpickled, wherever it came from.
ARCHIVE_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
FILE_SHA256 = {
    "data_batch_1": "54636561a3ce25bd3e19253c6b0d8538147b0ae398331ac4a2d86c6d987368cd",
    "data_batch_2": "766b2cef9fbc745cf056b3152224f7cf77163b330ea9a15f9392beb8b89bc5a8",
    "data_batch_3": "0f00d98ebfb30b3ec0ad19f9756dc2630b89003e10525f5e148445e82aa6a1f9",
    "data_batch_4": "3f7bb240661948b8f4d53e36ec720d8306f5668bd0071dcb4e6c947f78e9682b",
    "data_batch_5": "d91802434d8376bbaeeadf58a737e3a1b12ac839077e931237e0dcd43adcb154",
    "test_batch": "f53d8d457504f7cff4ea9e021afcf0e0ad8e24a91f3fc42091b8adef61157831",
}
MEAN = np.array([125.307, 122.961, 113.8575], np.float32) / 255
STD = np.array([51.5865, 50.847, 51.255], np.float32) / 255

HYP = dict(
    epochs=9.9, batch_size=1024, lr=11.5, momentum=0.85, weight_decay=0.0153,
    bias_scaler=64.0, label_smoothing=0.2, whiten_bias_epochs=3,
    widths=(64, 256, 256), bn_momentum=0.6, scaling_factor=1 / 9,
    translate=2, tta_level=2,
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _check(path, expected):
    got = _sha256(path)
    if got != expected:
        raise RuntimeError(f"{path}: SHA-256 {got}, expected {expected}; delete it and "
                           "rerun to download it again")


def load_cifar10(root=DATA):
    """(train_x, train_y, test_x, test_y): uint8 NHWC images, int32 labels."""
    d = os.path.join(root, "cifar-10-batches-py")
    names = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
    if not all(os.path.exists(os.path.join(d, n)) for n in names):
        os.makedirs(root, exist_ok=True)
        tgz = os.path.join(root, "cifar-10-python.tar.gz")
        if not os.path.exists(tgz):
            print(f"downloading {URL} to {root}", file=sys.stderr)
            urllib.request.urlretrieve(URL, tgz + ".part")
            os.replace(tgz + ".part", tgz)
        _check(tgz, ARCHIVE_SHA256)
        with tarfile.open(tgz) as t:
            t.extractall(root, filter="data")

    def read(names):
        xs, ys = [], []
        for n in names:
            _check(os.path.join(d, n), FILE_SHA256[n])
            with open(os.path.join(d, n), "rb") as f:
                b = pickle.load(f, encoding="bytes")
            xs.append(b[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            ys.append(np.array(b[b"labels"], np.int32))
        return np.concatenate(xs), np.concatenate(ys)

    tx, ty = read(names[:5])
    vx, vy = read(names[5:])
    return tx, ty, vx, vy
