import os
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Dataset options
__C.DATASET = edict()
__C.DATASET.TRAIN_BATCH_SIZE = 4
__C.DATASET.TEST_BATCH_SIZE = 1
__C.DATASET.POINT_NUM = 1024
__C.DATASET.UNSEEN = False
__C.DATASET.NOISE_TYPE = 'clean'
__C.DATASET.ROT_MAG = 45.0
__C.DATASET.TRANS_MAG = 0.5
__C.DATASET.PARTIAL_P_KEEP = [0.7, 0.7]

# Model options
__C.MODEL = edict()
__C.MODEL.NEIGHBORSNUM = 20
__C.MODEL.FEATURE_EMBED_CHANNEL = 512
__C.MODEL.SKADDCR = False

# Model name and dataset name
__C.MODEL_NAME = 'UTOPIC'
__C.DATASET_NAME = 'ModelNet40'
__C.DATASET_FULL_NAME = 'modelnet40_2048'

# Output path (for checkpoints, running logs and visualization results)
__C.OUTPUT_PATH = ''

# num of dataloader processes
__C.DATALOADER_NUM = 0

# The step of iteration to print running statistics.
__C.STATISTIC_STEP = 5

# random seed used for data2d loading
__C.RANDOM_SEED = 123

# Parallel GPU indices ([0] for single GPU)
__C.GPU = 0

# Training options
__C.TRAIN = edict()

# Training start epoch. If not 0, will be resumed from checkpoint.
__C.TRAIN.START_EPOCH = 0

# Total epochs
__C.TRAIN.NUM_EPOCHS = 200

# Start learning rate
__C.TRAIN.OPTIM = ''
__C.TRAIN.LR = 0.01

# SGD momentum
__C.TRAIN.MOMENTUM = 0.9

# Learning rate decay
__C.TRAIN.LR_DECAY = 0.1

# Learning rate decay step (in epochs)
__C.TRAIN.LR_STEP = [10, 20]

# Evaluation options
__C.EVAL = edict()

# Evaluation epoch number
__C.EVAL.EPOCH = 0


def lcm(x, y):
    """
    Compute the least common multiple of x and y. This function is used for running statistics.
    """
    greater = max(x, y)
    while True:
        if (greater % x == 0) and (greater % y == 0):
            lcm = greater
            break
        greater += 1
    return lcm


def get_output_dir(model, dataset):
    """
    Return the directory where experimental artifacts are placed.
    :param model: model name
    :param dataset: dataset name
    :return: output path (checkpoint and log), visual path (visualization images)
    """
    outp_path = os.path.join('output', '{}_{}'.format(model, dataset))
    return outp_path


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]),
                                                           type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d.keys()
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d.keys()
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value
