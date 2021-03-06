import os
import yaml
import time
import argparse
import importlib
from torch.backends import cudnn
from shutil import copy2
from pprint import pprint

def get_args():
    # command line args
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('config', type=str,
                        help='The configuration file.')

    parser.add_argument('--pretrained', default=None, type=str,
                        help="Pretrained cehckpoint")

    parser.add_argument('--eval_split', default='val', type=str,
                        help="The split to be evaluated.")
    args = parser.parse_args()

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    # parse config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2namespace(config)

    # #  Create log_name
    cfg_file_name = os.path.splitext(os.path.basename(args.config))[0]
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    config.log_name = "logs/%s_val_%s" % (cfg_file_name, run_time)
    config.save_dir = "logs/%s_val_%s" % (cfg_file_name, run_time)
    config.log_dir = "logs/%s_val_%s" % (cfg_file_name, run_time)
    os.makedirs(config.log_dir + '/config')
    copy2(args.config, config.log_dir + '/config')
    return args, config


def main_worker(cfg, args):
    cudnn.benchmark = True
    data_lib = importlib.import_module(cfg.data.type)
    loaders = data_lib.get_data_loaders(cfg.data, args)
    test_loader = loaders['test_loader']
    tester_lib = importlib.import_module(cfg.tester.type)

    tester = tester_lib.Tester(cfg, args)

    tester.resume(args.pretrained)
    val_info = tester.validate(test_loader, epoch=-1)

    print("Test done:")
    pprint(val_info)


if __name__ == '__main__':
    # command line args
    args, cfg = get_args()

    print("Arguments:")
    print(args)

    print("Configuration:")
    print(cfg)

    main_worker(cfg, args)
