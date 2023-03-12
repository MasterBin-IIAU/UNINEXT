from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)



external_dir = os.path.join(os.path.dirname(__file__), "..", "external")
if external_dir not in sys.path:
    add_path(external_dir)
