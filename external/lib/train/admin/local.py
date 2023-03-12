from unicorn.data import get_unicorn_datadir
import os

class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = os.path.join(get_unicorn_datadir(), '..')    # Base directory for saving network checkpoints.
        self.coco_dir = os.path.join(get_unicorn_datadir(), 'COCO')
        # VOS
        self.davis_dir = os.path.join(get_unicorn_datadir(), 'DAVIS')
        self.davis16_dir = os.path.join(get_unicorn_datadir(), 'DAVIS')
        self.youtubevos_dir = os.path.join(get_unicorn_datadir(), 'ytbvos18') # youtube-vos 2018 val
