import sys
from argparse import ArgumentParser
from distutils import command

from preprocess.dataset import get_dataloader, DatasetType
from preprocess.process import process_data
from runner.evaluate import run_evalaute
from runner.predict import run_predict
from runner.train import train, get_device


if __name__ == '__main__':
    arg_parser = ArgumentParser(usage='python main.py command [--epoch EPOCH]', description='入口脚本')
    arg_parser.print_help()
    arg_parser.add_argument('command', choices=['train', 'predict', 'evaluate', 'preprocess','server'])
    args = arg_parser.parse_args()

    command = args.command

    if command == 'train':
        train()
    elif command == 'predict':
        run_predict()
    elif command == 'evaluate':
        run_evalaute()
    elif command == 'preprocess':
        process_data()

    elif command == 'server':
        from web.app import run_app
        run_app()