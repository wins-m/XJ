"""
(created by swmao on June 12th)
In shell:
    ```sh
    python script.py --gpus=0,1,2 --batch-size=10

    ```
通过这个方法还能指定命令的帮助信息。
具体请看API文档：https://docs.python.org/2/library/argparse.html

"""
from argparse import ArgumentParser
parser = ArgumentParser(description='manual to this script')
parser.add_argument('--gpus', type=str, default = None)
parser.add_argument('--batch-size', type=int, default=32)
args = parser.parse_args()
print(args.gpus)
print(args.batch_size)

