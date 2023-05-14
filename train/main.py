import sys

sys.path.append('.')

from train.trainer.predictor.trainer_finetune import FinetuneTrainer

from train.configs.finetune_args import args as finetune_args



if __name__ == '__main__':
    trainer = FinetuneTrainer(finetune_args)
    trainer.train(task_id=0)
    # trainer.evaluate_continual_learning_metrics()
