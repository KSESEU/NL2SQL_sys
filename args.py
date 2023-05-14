import argparse

def init_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_name', default="spider", type=str)
    arg_parser.add_argument('--data_path', default="./database/{}", type=str)
    arg_parser.add_argument('--model_path', default="ckpt/finetune_t5-small/spider_perm_1/task_0", type=str)
    arg_parser.add_argument('--prop_data_path', default="datasets/spider/original_data", type=str)
    args = arg_parser.parse_args()
    return args
