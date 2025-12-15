"""
WeedSwin Training Script
This script trains the WeedSwin model for weed detection and classification.
"""
from mmengine.config import Config
from mmengine.runner import Runner

def main():
    # Load configuration file
    cfg = Config.fromfile('./configs/weedswin.py')

    # Set work directory for saving logs and checkpoints
    cfg.work_dir = "work_dirs/weedswin"

    # Build the runner from config
    runner = Runner.from_cfg(cfg)

    # Start training
    runner.train()

if __name__ == '__main__':
    main()
