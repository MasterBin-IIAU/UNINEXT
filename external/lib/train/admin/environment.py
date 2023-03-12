import importlib
import os
from collections import OrderedDict


def create_default_local_file_train(workspace_dir, data_dir):
    path = os.path.join(os.path.dirname(__file__), 'local.py')

    empty_str = '\'\''
    default_settings = OrderedDict({
        'workspace_dir': workspace_dir,
        'tensorboard_dir': os.path.join(workspace_dir, 'tensorboard'),  # Directory for tensorboard files.
        'pretrained_networks': os.path.join(workspace_dir, 'pretrained_networks'),
        'coco_dir': os.path.join(data_dir, 'COCO2017'),
        'davis_dir': os.path.join(data_dir, 'DAVIS'),
        'youtubevos_dir': os.path.join(data_dir, 'ytbvos18'),
        })

    comment = {'workspace_dir': 'Base directory for saving network checkpoints.',
               'tensorboard_dir': 'Directory for tensorboard files.'}

    with open(path, 'w') as f:
        f.write('class EnvironmentSettings:\n')
        f.write('    def __init__(self):\n')

        for attr, attr_val in default_settings.items():
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]
            if comment_str is None:
                if attr_val == empty_str:
                    f.write('        self.{} = {}\n'.format(attr, attr_val))
                else:
                    f.write('        self.{} = \'{}\'\n'.format(attr, attr_val))
            else:
                f.write('        self.{} = \'{}\'    # {}\n'.format(attr, attr_val, comment_str))


def env_settings():
    env_module_name = 'lib.train.admin.local'
    try:
        env_module = importlib.import_module(env_module_name)
        return env_module.EnvironmentSettings()
    except:
        env_file = os.path.join(os.path.dirname(__file__), 'local.py')
        raise RuntimeError(
            'YOU HAVE NOT SETUP YOUR local.py!!!\n Go to "{}" and set all the paths you need. Then try to run again.'.format(
                env_file))
