import argparse


class DefaultConfig(object):
    # 这个是可有可无的，尽量不使用这里的参数
    pass


class Args(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def get_all_args(self):
        args = self.parser.parse_args()

        return args


def main():
    args = Args().get_all_args()


if __name__ == '__main__':
    main()
