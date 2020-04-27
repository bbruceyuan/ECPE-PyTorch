import random
import numpy as np
import os

import torch
import torch.backends
import torch.backends.cudnn

from args import Args
from log_config import LOG_CONFIG
import trainers

import logging
import logging.config


def set_log():
    logging.config.dictConfig(LOG_CONFIG)
    logger = logging.getLogger(__name__)
    return logger


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # 这个好像我也不太会用
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)


def main():
    logger = set_log()
    logger.info('程序运行开始')

    args = Args().get_all_args()
    set_seed(args)
    # 进行模型的训练
    # 根据参数选择 trainer
    Trainer = getattr(trainers, args.trainer_name)
    trainer = Trainer(args)
    args.do_train = True
    if args.do_train:
        trainer.train()

    # 在做 test 之前，应该需要 load 以前保存的最佳模型
    if args.do_test:
        trainer.test()


def test():
    pass


if __name__ == '__main__':
    main()
