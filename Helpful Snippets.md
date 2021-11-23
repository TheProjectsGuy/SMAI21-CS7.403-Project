# Helpful Snippets

These are chunks of code that can be re-used, but aren't formally documented in this repository. These snippets could be useful in many places.

## Table of contents

- [Helpful Snippets](#helpful-snippets)
    - [Table of contents](#table-of-contents)
    - [Include Module in PYTHONPATH](#include-module-in-pythonpath)
    - [TensorFlow System test](#tensorflow-system-test)

## Include Module in PYTHONPATH

The `PYTHONPATH` variable must contain the path to this repo (which is a module in itself) in order to run code. To do that, the following gimmick is used

```py
# This is done to add the entire repository as if it's a module
import os
import sys
from pathlib import Path
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
except NameError:
    print("WARN: __file__ not found, trying local")
    dir_name = os.path.abspath('')
# Top project directory (for accessing everything)
pkg_path = str(Path(dir_name).parent.parent.parent)
# Add to path
if pkg_path not in sys.path:
    sys.path.append(pkg_path)
    print("Added package path")
```

The alternative is to define the repository as a python package (not recommended for a simple project).

## TensorFlow System test

It's sometimes useful to know properties of system before starting anything. The following snippet gives some useful information

```py
# TensorFlow
print(f"Tensorflow Version: {tf.__version__}")
devs = tf.config.list_physical_devices()
for dev in devs:
    print(f"Found device: {dev}")
    if dev.device_type == "GPU":
        print("CUDA acceleration available")
```
