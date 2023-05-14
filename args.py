import argparse

def init_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_name', default="spider", type=str)
    # arg_parser.add_argument('--data_name', default="esql", type=str)
    arg_parser.add_argument('--data_path', default="./database/{}", type=str)
    arg_parser.add_argument('--sample_path', default="./datasets/{}/original_data/train_seq2seq.jsonl", type=str)
    arg_parser.add_argument('--model_path', default="ckpt/finetune_t5-small/spider/task_0", type=str)
    # arg_parser.add_argument('--model_path', default="ckpt/esql_t5-small-chinese_final_20epoch/esql/task_0", type=str)
    arg_parser.add_argument('--prop_data_path', default="datasets/{}/original_data", type=str)
    # arg_parser.add_argument('--prop_data_path', default="datasets/esql/original_data", type=str)
    arg_parser.add_argument('--use_cn_translate', default=1, type=int)
    arg_parser.add_argument('--use_direct', default=1, type=int)
    args = arg_parser.parse_args()
    return args
