"""
WeedSwin Testing Script
This script tests the trained WeedSwin model on test dataset.
"""
import argparse
from mmengine.config import Config
from mmengine.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser(description='Test WeedSwin detector')
    parser.add_argument('--config', default='./configs/weedswin.py', help='test config file path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint file path')
    parser.add_argument('--work-dir', default='work_dirs/weedswin_test', help='directory to save test results')
    parser.add_argument('--show', action='store_true', help='show prediction results')
    parser.add_argument('--show-dir', help='directory where painted images will be saved')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load configuration
    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    cfg.load_from = args.checkpoint

    # Build runner and run testing
    runner = Runner.from_cfg(cfg)
    runner.test()

if __name__ == '__main__':
    main()
