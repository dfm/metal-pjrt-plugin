"""CIFAR-10 loading and airbench94's hyperparameters, shared by airbench.py
(JAX) and torch_baseline.py (PyTorch); numpy only.

CIFAR-10 (Krizhevsky, "Learning Multiple Layers of Features from Tiny
Images", 2009) is downloaded once from the authors' site into
~/.cache/metal-pjrt-examples; it is not part of this repository.
"""
import os, pickle, sys, tarfile, urllib.request

import numpy as np

DATA = os.path.expanduser("~/.cache/metal-pjrt-examples")
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MEAN = np.array([125.307, 122.961, 113.8575], np.float32) / 255
STD = np.array([51.5865, 50.847, 51.255], np.float32) / 255

HYP = dict(
    epochs=9.9, batch_size=1024, lr=11.5, momentum=0.85, weight_decay=0.0153,
    bias_scaler=64.0, label_smoothing=0.2, whiten_bias_epochs=3,
    widths=(64, 256, 256), bn_momentum=0.6, scaling_factor=1 / 9,
    translate=2, tta_level=2,
)


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
        with tarfile.open(tgz) as t:
            t.extractall(root, filter="data")

    def read(names):
        xs, ys = [], []
        for n in names:
            with open(os.path.join(d, n), "rb") as f:
                b = pickle.load(f, encoding="bytes")
            xs.append(b[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            ys.append(np.array(b[b"labels"], np.int32))
        return np.concatenate(xs), np.concatenate(ys)

    tx, ty = read(names[:5])
    vx, vy = read(names[5:])
    return tx, ty, vx, vy
