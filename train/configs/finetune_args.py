import argparse

parser = argparse.ArgumentParser('')
parser.add_argument('--backbone_plm', type=str, default='t5-small-chinese')
parser.add_argument('--model_name_or_path', type=str, default='')
parser.add_argument('--total_output_dir', type=str, default='ckpt/esql')
parser.add_argument('--total_dataset_dir', type=str, default='train/data_train')
parser.add_argument('--dataset', type=str, default='esql')
parser.add_argument('--per_device_train_batch_size', type=int, default=12)
parser.add_argument('--per_device_eval_batch_size', type=int, default=12)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('--metric_for_best_model', type=str, default='exact_match')
parser.add_argument('--greater_is_better', type=bool, default=True)
parser.add_argument('--max_source_length', type=int, default=512)
parser.add_argument('--max_target_length', type=int, default=256)
parser.add_argument('--overwrite_output_dir', type=bool, default=True)
parser.add_argument('--do_train', type=bool, default=True)
parser.add_argument('--do_predict', type=bool, default=True)
parser.add_argument('--predict_with_generate', type=bool, default=True)
parser.add_argument('--lr_scheduler_type', type=str, default='linear')
parser.add_argument('--label_smoothing_factor', type=float, default=0.0)
parser.add_argument('--warmup_ratio', type=float, default=0.0)
parser.add_argument('--warmup_steps', type=int, default=0)
parser.add_argument('--num_beams', type=int, default=4)
parser.add_argument('--optim', type=str, default='adafactor')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--adam_epsilon', type=float, default=1e-06)
parser.add_argument('--load_best_model_at_end', type=bool, default=True)
parser.add_argument('--num_train_epochs', type=int, default=1000)
parser.add_argument('--save_strategy', type=str, default='steps')
parser.add_argument('--save_total_limit', type=int, default=1)
parser.add_argument('--evaluation_strategy', type=str, default='steps')

args = parser.parse_args()

if args.backbone_plm == 't5-small-lm-adapt':
    args.total_output_dir += '_t5-small'
elif args.backbone_plm == 't5-base-lm-adapt':
    args.total_output_dir += '_t5-base'
elif args.backbone_plm == 't5-large-lm-adapt':
    args.total_output_dir += '_t5-large'
elif args.backbone_plm == 't5-small-chinese':
    args.total_output_dir += '_t5-small-chinese'
else:
    raise NotImplementedError


args.output_dir = f'{args.total_output_dir}/{args.dataset}'
args.dataset_dir = f'{args.total_dataset_dir}/{args.dataset}'
