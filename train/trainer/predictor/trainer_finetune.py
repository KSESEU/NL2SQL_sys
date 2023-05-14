import json
import logging
import os
import shutil
import sys
import datasets
import nltk
import torch
from torch.utils.data import Dataset
import math
import time
import transformers
from filelock import FileLock
from typing import Optional, List, NamedTuple
from dataclasses import dataclass, field
import numpy as np

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    EarlyStoppingCallback
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import speed_metrics
from transformers.trainer_seq2seq import Seq2SeqTrainer

from train.trainer.predictor.trainer_predictor import PredictorTrainer
from train.train_utils.spider.evaluation import evaluate as spider_evaluate


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                    "the model's position embeddings."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
                    "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
                    "Useful for multilingual models like mBART where the first generated token"
                    "needs to be the target language token (Usually it is the target language token)"
        },
    )

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    total_output_dir: str = field(
        default=None
    )
    dataset: str = field(
        default=None
    )
    total_dataset_dir: str = field(
        default=None
    )
    dataset_dir: str = field(
        default=None
    )


class EvalPrediction(NamedTuple):
    predictions: List[str]
    gold_sql: List[str]
    gold_db: List[str]


class SubTrainer(Seq2SeqTrainer):
    def __init__(self, dataset_type: str, **kwargs):
        super().__init__(**kwargs)
        self.compute_metrics = self._compute_metrics
        self.output_dir = self.args.output_dir
        self.dataset_type = dataset_type
        if self.dataset_type == 'spider' or self.dataset_type == 'sparc':
            self.etype = 'all'
        elif self.dataset_type == 'wikisql':
            self.etype = 'match'

    def _post_process(self, dataset: Dataset, predictions: np.ndarray):
        gold_sql = dataset['sql']
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        gold_db = dataset['db_id']
        interaction_ids = dataset['interaction_id']

        if self.dataset_type == 'sparc':
            pre_ia_id, g_ia_db = None, None
            g_ia_one, p_ia_one = [], []
            gold_interactions, pred_interactions, gold_interaction_dbs = [], [], []
            for i, ia_id in enumerate(interaction_ids):
                if ia_id != pre_ia_id and pre_ia_id is not None:
                    gold_interactions.append(g_ia_one)
                    pred_interactions.append(p_ia_one)
                    gold_interaction_dbs.append(g_ia_db)
                    g_ia_one, p_ia_one = [], []
                    g_ia_db = None
                else:
                    g_ia_one.append(gold_sql[i])
                    p_ia_one.append(predictions[i])
                    g_ia_db = gold_db[i]
                pre_ia_id = ia_id

            predictions = pred_interactions
            gold_sql = gold_interactions
            gold_db = gold_interaction_dbs

        return EvalPrediction(predictions=predictions, gold_sql=gold_sql, gold_db=gold_db)

    def _compute_metrics(self, eval_prediction: EvalPrediction, etype: str):
        predictions, gold_sql, gold_db = eval_prediction
        if self.dataset_type == 'spider':
            evaluation_method = spider_evaluate
        else:
            raise NotImplementedError

        if etype == 'match':
            match_score, eval_results = evaluation_method(gold_sql=gold_sql, gold_db=gold_db,
                                                          predict=predictions, etype='match')
            metrics = {'eval_exact_match': match_score}
        else:
            exec_score, match_score, eval_results = evaluation_method(gold_sql=gold_sql, gold_db=gold_db,
                                                                      predict=predictions, etype='all')
            metrics = {'eval_exec': exec_score, 'eval_exact_match': match_score}
        return metrics, eval_results

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "test",
            **gen_kwargs
    ):

        gen_kwargs = gen_kwargs.copy()
        gen_kwargs["max_length"] = 256
        self._gen_kwargs = gen_kwargs

        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self.compute_metrics = compute_metrics

        total_batch_size = self.args.eval_batch_size * self.args.world_size

        eval_preds = self._post_process(self.eval_dataset, output.predictions)
        metrics, eval_results = compute_metrics(eval_preds, etype=self.etype)

        with open(f'{self.output_dir}/eval_log.jsonl', 'a', encoding='utf-8') as writer:
            writer.write(json.dumps(metrics) + '\n')

        output.metrics.update(metrics)

        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
            self,
            test_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            **gen_kwargs
    ):
        gen_kwargs = gen_kwargs.copy()
        gen_kwargs["max_length"] = 256
        self._gen_kwargs = gen_kwargs

        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(test_dataset)
        start_time = time.time()

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Prediction",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self.compute_metrics = compute_metrics

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        preds = self._post_process(test_dataset, output.predictions)
        pred_metric, eval_results = compute_metrics(preds, etype=self.etype)
        output.metrics.update(pred_metric)

        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output, eval_results


class FinetuneTrainer(PredictorTrainer):
    def __init__(self, args):
        super().__init__(args)
        set_seed(self.args.seed)
        self.trainer = None
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

        print(args)

        self.model_args, self.data_args, self.training_args = parser.parse_dict(vars(args))

        try:
            nltk.data.find("tokenizers/punkt")
        except (LookupError, OSError):
            if is_offline_mode():
                raise LookupError(
                    "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
                )
            with FileLock(".lock") as lock:
                nltk.download("punkt", quiet=True)

        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        log_level = self.training_args.get_process_log_level()
        self.logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.logging.set_verbosity(log_level)
        transformers.logging.enable_default_handler()
        transformers.logging.enable_explicit_format()

        self.logger.warning(
            f"Process rank: {self.training_args.local_rank}, device: {self.training_args.device}, "
            f"n_gpu: {self.training_args.n_gpu}, distributed training: {bool(self.training_args.local_rank != -1)}, "
            f"16-bits training: {self.training_args.fp16}"
        )

    def prepare_raw_datasets(self, task_id: int):
        data_files = {
            'train': f'{self.args.dataset_dir}/task_{task_id}/train_seq2seq.jsonl',
            'validation': f'{self.args.dataset_dir}/task_{task_id}/dev_seq2seq.jsonl',
            'test': f'{self.args.dataset_dir}/task_{task_id}/test_seq2seq.jsonl'
        }

        raw_datasets = {}
        for ds_name in data_files.keys():
            raw_datasets[ds_name] = {'text': [], 'sql': [], 'db_id': [], 'dataset_type': [], 'interaction_id': []}
            with open(data_files[ds_name], 'r', encoding='utf-8') as reader:
                for line in reader.readlines():
                    ex = json.loads(line)
                    raw_datasets[ds_name]['text'].append(ex['text'])
                    raw_datasets[ds_name]['sql'].append(ex['sql'])
                    raw_datasets[ds_name]['db_id'].append(ex['example']['db_id'])
                    if 'combine' in self.args.dataset_dir:
                        raw_datasets[ds_name]['dataset_type'].append(ex['example']['dataset_type'])
                        raw_datasets[ds_name]['interaction_id'].append(None)
                    elif 'spider' in self.args.dataset_dir:
                        raw_datasets[ds_name]['dataset_type'].append('spider')
                        raw_datasets[ds_name]['interaction_id'].append(None)
                    elif 'sparc' in self.args.dataset_dir:
                        raw_datasets[ds_name]['dataset_type'].append('sparc')
                        raw_datasets[ds_name]['interaction_id'].append(ex['example']['interaction_id'])
                    elif 'esql' in self.args.dataset_dir:
                        raw_datasets[ds_name]['dataset_type'].append('wikisql')
                        raw_datasets[ds_name]['interaction_id'].append(None)
                    else:
                        raise NotImplementedError

            raw_datasets[ds_name] = datasets.Dataset.from_dict(raw_datasets[ds_name])

        return raw_datasets

    def train(self, task_id: int):
        torch.cuda.empty_cache()
        if task_id == 0:
            model_name_or_path = f'train/plms/{self.args.backbone_plm}'
        else:
            model_name_or_path = f'{self.args.output_dir}/task_{task_id - 1}'

        self.training_args.output_dir = f'{self.args.output_dir}/task_{task_id}'

        last_checkpoint = None

        raw_datasets = self.prepare_raw_datasets(task_id)

        config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            use_fast=self.model_args.use_fast_tokenizer,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )

        tokenizer.truncation_side = 'left'
        tokenizer.add_tokens(['<', '<='])

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )

        model.resize_token_embeddings(len(tokenizer))

        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        if (
                hasattr(model.config, "max_position_embeddings")
                and model.config.max_position_embeddings < self.data_args.max_source_length
        ):
            if self.model_args.resize_position_embeddings is None:
                self.logger.warning(
                    f"Increasing the model's number of position embedding vectors from "
                    f"{model.config.max_position_embeddings} to {self.data_args.max_source_length}."
                )
                model.resize_position_embeddings(self.data_args.max_source_length)
            elif self.model_args.resize_position_embeddings:
                model.resize_position_embeddings(self.data_args.max_source_length)
            else:
                raise ValueError(
                    f"`--max_source_length` is set to {self.data_args.max_source_length}, but the model only has "
                    f"{model.config.max_position_embeddings} position encodings. Consider either reducing "
                    f"`--max_source_length` to {model.config.max_position_embeddings} or to automatically resize "
                    f"the model's position encodings by passing `--resize_position_embeddings`."
                )

        prefix = self.data_args.source_prefix if self.data_args.source_prefix is not None else ""

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = raw_datasets['test'].column_names
        text_column, summary_column = column_names[0], column_names[1]

        # Temporarily set max_target_length for training.
        max_target_length = self.data_args.max_target_length
        padding = self.data_args.pad_to_max_length

        if self.training_args.label_smoothing_factor > 0 and \
                not hasattr(model, "prepare_decoder_input_ids_from_labels"):
            self.logger.warning(
                f"label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined "
                f"for `{model.__class__.__name__}`. "
                f"This will lead to loss being calculated twice and will take up more memory"
            )

        def preprocess_function(examples):
            # remove pairs where at least one record is None
            inputs, targets = [], []
            for i in range(len(examples[text_column])):
                if examples[text_column][i] is not None and examples[summary_column][i] is not None:
                    inputs.append(examples[text_column][i])
                    targets.append(examples[summary_column][i])

            inputs = [prefix + inp for inp in inputs]
            model_inputs = tokenizer(inputs, max_length=self.data_args.max_source_length, padding=padding,
                                     truncation=True)
            # Set up the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
            model_inputs["labels"] = labels["input_ids"]

            return model_inputs

        if self.training_args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets["train"]

            num_examples = len(train_dataset)
            steps_per_epoch = math.ceil(
                num_examples / (
                        self.training_args.per_device_train_batch_size * self.training_args.gradient_accumulation_steps))

            self.training_args.eval_steps = steps_per_epoch * 5
            self.training_args.save_steps = self.training_args.eval_steps
            self.training_args.eval_delay = steps_per_epoch * 50

            with self.training_args.main_process_first(desc="train dataset map pre-processing"):
                train_dataset = train_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    # remove_columns=column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )

        if self.training_args.do_eval:
            max_target_length = self.data_args.val_max_target_length
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            if self.data_args.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(range(self.data_args.max_eval_samples))
            with self.training_args.main_process_first(desc="validation dataset map pre-processing"):
                eval_dataset = eval_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    # remove_columns=column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )

        if self.training_args.do_predict:
            max_target_length = self.data_args.val_max_target_length
            if "test" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            predict_dataset = raw_datasets["test"]
            with self.training_args.main_process_first(desc="prediction dataset map pre-processing"):
                predict_dataset = predict_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    # remove_columns=column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                )

            # Data collator
        label_pad_token_id = -100 if self.data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if self.training_args.fp16 else None,
        )

        # Initialize our Trainer

        trainer_args = {
            'model': model,
            'args': self.training_args,
            'train_dataset': train_dataset if self.training_args.do_train else None,
            'eval_dataset': eval_dataset if self.training_args.do_eval else None,
            'tokenizer': tokenizer,
            'data_collator': data_collator,
            'compute_metrics': None,
            'callbacks': [EarlyStoppingCallback(early_stopping_patience=5)],
        }


        if self.training_args.dataset.startswith('combine'):
            if raw_datasets['train']['dataset_type'][0] == 'wikisql':
                dataset_type = 'wikisql'
                trainer = SubTrainer(dataset_type=dataset_type, **trainer_args)
            elif raw_datasets['train']['dataset_type'][0] == 'spider':
                dataset_type = 'spider'
                trainer = SubTrainer(dataset_type=dataset_type, **trainer_args)
            else:
                raise NotImplementedError
        elif self.training_args.dataset.startswith('spider'):
            dataset_type = 'spider'
            trainer = SubTrainer(dataset_type=dataset_type, **trainer_args)
        elif raw_datasets['train']['dataset_type'][0] == 'sparc':
            dataset_type = 'sparc'
            trainer = SubTrainer(dataset_type, **trainer_args)
        elif raw_datasets['train']['dataset_type'][0] == 'wikisql':
            dataset_type = 'wikisql'
            trainer = SubTrainer(dataset_type=dataset_type, **trainer_args)
        else:
            raise NotImplementedError

        if self.training_args.do_train:
            checkpoint = None
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics
            max_train_samples = (
                self.data_args.max_train_samples if self.data_args.max_train_samples is not None else len(
                    train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # Evaluation
        max_length = (
            self.training_args.generation_max_length
            if self.training_args.generation_max_length is not None
            else self.data_args.val_max_target_length
        )
        num_beams = self.data_args.num_beams if self.data_args.num_beams is not None else self.training_args.generation_num_beams

        if self.training_args.do_predict:
            self.logger.info("*** Predict ***")
            output, eval_results = trainer.predict(
                predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
            )

            metrics = output.metrics
            max_predict_samples = (
                self.data_args.max_predict_samples if self.data_args.max_predict_samples is not None else len(
                    predict_dataset)
            )
            metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
            predictions = tokenizer.batch_decode(
                output.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]

            # save metrics & predict results
            output_spider_result_file = os.path.join(self.training_args.output_dir, "eval_results.txt")
            with open(output_spider_result_file, "w", encoding='utf-8') as writer:
                writer.write("\n".join(eval_results))
            output_prediction_file = os.path.join(self.training_args.output_dir, "generated_predictions.txt")
            with open(output_prediction_file, "w", encoding='utf-8') as writer:
                writer.write("\n".join(predictions))

        # delete checkpoints
        for name in os.listdir(self.training_args.output_dir):
            if os.path.isdir(os.path.join(self.training_args.output_dir, name)):
                shutil.rmtree(os.path.join(self.training_args.output_dir, name))

    def evaluate_continual_learning_metrics(self):
        def get_task_size(data_dir: str, task_num: int):
            task_size_list = []
            for i in range(task_num):
                with open(f'{data_dir}/task_{i}/test_seq2seq.jsonl', 'r') as reader:
                    size = len(reader.readlines())
                task_size_list.append(size)
            return task_size_list

        def compute_acc(output_dir: str, task_id: int, data_dir: str, task_num: int):
            task_size_list = get_task_size(data_dir, task_num)
            acc_a, acc_w = 0, 0
            for t_id in range(task_id + 1):
                with open(f'{output_dir}/metrics/task_{task_id}/task_{t_id}/eval_results.txt', 'r') as reader:
                    for line in reader.readlines():
                        if line.startswith('exact match'):
                            acc = float(line.strip()[-5:])
                            break

                    acc_a += acc
                    acc_w += acc * task_size_list[t_id]

            acc_a /= task_id + 1
            acc_w /= sum(task_size_list[:task_id + 1])

            return acc_a, acc_w

        def compute_bwt(output_dir, task_id):
            bwt = 0
            for t_id in range(task_id):
                with open(f'{output_dir}/metrics/task_{task_id}/task_{t_id}/eval_results.txt', 'r') as reader:
                    for line in reader.readlines():
                        if line.startswith('exact match'):
                            backward_acc = float(line.strip()[-5:])
                            break
                with open(f'{output_dir}/metrics/task_{t_id}/task_{t_id}/eval_results.txt', 'r') as reader:
                    for line in reader.readlines():
                        if line.startswith('exact match'):
                            original_acc = float(line.strip()[-5:])
                            break
                bwt += (backward_acc - original_acc)
            bwt /= task_id
            return bwt

        def compute_fwt(output_dir, task_id):
            fwt = 0
            for t_id in range(task_id + 1):
                with open(f'{output_dir}/metrics/task_{t_id}/task_{str(t_id + 1)}/eval_results.txt', 'r') as reader:
                    random_acc = 0.0
                    for line in reader.readlines():
                        if line.startswith('exact match'):
                            forward_acc = float(line.strip()[-5:])
                            break
                fwt += (forward_acc - random_acc)
            fwt /= (task_id + 1)
            return fwt

        def prepare_raw_datasets_for_cl_evaluation(dataset_dir: str, cur_task_id: int, to_eval_task_id: int):
            if cur_task_id > to_eval_task_id:
                mode = 'backward'
            elif cur_task_id < to_eval_task_id:
                mode = 'forward'
            else:
                raise NotImplementedError

            if mode in os.listdir(f'{dataset_dir}/task_{cur_task_id}'):
                data_files = {
                    'test': f'{dataset_dir}/task_{cur_task_id}/{mode}/task_{to_eval_task_id}_test_seq2seq.jsonl'
                }
            else:
                data_files = {
                    'test': f'{dataset_dir}/task_{to_eval_task_id}//test_seq2seq.jsonl'
                }

            raw_datasets = {}
            for ds_name in data_files.keys():
                raw_datasets[ds_name] = {
                    'text': [],
                    'sql': [],
                    'db_id': [],
                    'dataset_type': [],
                    'interaction_id': []
                }
                with open(data_files[ds_name], 'r', encoding='utf-8') as reader:
                    for line in reader.readlines():
                        ex = json.loads(line)
                        raw_datasets[ds_name]['text'].append(ex['text'])
                        raw_datasets[ds_name]['sql'].append(ex['sql'])
                        raw_datasets[ds_name]['db_id'].append(ex['example']['db_id'])
                        if 'combine' in self.args.dataset_dir:
                            raw_datasets[ds_name]['dataset_type'].append(ex['example']['dataset_type'])
                        elif 'spider' in self.args.dataset_dir:
                            raw_datasets[ds_name]['dataset_type'].append('spider')
                        elif 'sparc' in self.args.dataset_dir:
                            raw_datasets[ds_name]['dataset_type'].append('sparc')
                            raw_datasets[ds_name]['interaction_id'].append(ex['example']['dataset_type']['interaction_id'])
                        else:
                            raise NotImplementedError
                raw_datasets[ds_name] = datasets.Dataset.from_dict(raw_datasets[ds_name])

            return raw_datasets

        def predict(cur_task_id: int, to_eval_task_id: int):
            model_name_or_path = f'{self.args.output_dir}/task_{cur_task_id}'

            raw_datasets = prepare_raw_datasets_for_cl_evaluation(self.args.dataset_dir, cur_task_id, to_eval_task_id)

            config = AutoConfig.from_pretrained(
                self.model_args.config_name if self.model_args.config_name else model_name_or_path,
                cache_dir=self.model_args.cache_dir,
                revision=self.model_args.model_revision,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_args.tokenizer_name if self.model_args.tokenizer_name else model_name_or_path,
                cache_dir=self.model_args.cache_dir,
                use_fast=self.model_args.use_fast_tokenizer,
                revision=self.model_args.model_revision,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )

            tokenizer.truncation_side = 'left'
            tokenizer.add_tokens(['<', '<='])

            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                from_tf=False,
                config=config,
                cache_dir=self.model_args.cache_dir,
                revision=self.model_args.model_revision,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )

            model.resize_token_embeddings(len(tokenizer))

            if model.config.decoder_start_token_id is None:
                raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

            if (
                    hasattr(model.config, "max_position_embeddings")
                    and model.config.max_position_embeddings < self.data_args.max_source_length
            ):
                if self.model_args.resize_position_embeddings is None:
                    self.logger.warning(
                        f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                        f"to {self.data_args.max_source_length}."
                    )
                    model.resize_position_embeddings(self.data_args.max_source_length)
                elif self.model_args.resize_position_embeddings:
                    model.resize_position_embeddings(self.data_args.max_source_length)
                else:
                    raise ValueError(
                        f"`--max_source_length` is set to {self.data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                        f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                        "resize the model's position encodings by passing `--resize_position_embeddings`."
                    )

            prefix = self.data_args.source_prefix if self.data_args.source_prefix is not None else ""

            # Preprocessing the datasets.
            # We need to tokenize inputs and targets.
            column_names = raw_datasets['test'].column_names
            text_column, summary_column = column_names[0], column_names[1]

            # Temporarily set max_target_length for training.
            max_target_length = self.data_args.max_target_length
            padding = self.data_args.pad_to_max_length

            if self.training_args.label_smoothing_factor > 0 and not hasattr(model,
                                                                             "prepare_decoder_input_ids_from_labels"):
                self.logger.warning(
                    "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                    f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
                )

            def preprocess_function(examples):
                # remove pairs where at least one record is None

                inputs, targets = [], []
                for i in range(len(examples[text_column])):
                    if examples[text_column][i] is not None and examples[summary_column][i] is not None:
                        inputs.append(examples[text_column][i])
                        targets.append(examples[summary_column][i])

                inputs = [prefix + inp for inp in inputs]
                model_inputs = tokenizer(inputs, max_length=self.data_args.max_source_length, padding=padding,
                                         truncation=True)

                # Setup the tokenizer for targets
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

                model_inputs["labels"] = labels["input_ids"]
                return model_inputs

            max_target_length = self.data_args.val_max_target_length
            if "test" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            predict_dataset = raw_datasets["test"]
            if self.data_args.max_predict_samples is not None:
                predict_dataset = predict_dataset.select(range(self.data_args.max_predict_samples))
            with self.training_args.main_process_first(desc="prediction dataset map pre-processing"):
                predict_dataset = predict_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    # remove_columns=column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                )

            # Data collator
            label_pad_token_id = -100 if self.data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
            data_collator = DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8 if self.training_args.fp16 else None,
            )

            trainer_args = {
                'model': model,
                'args': self.training_args,
                'tokenizer': tokenizer,
                'data_collator': data_collator,
                'compute_metrics': None,
                'callbacks': [EarlyStoppingCallback(early_stopping_patience=5)],
            }

            if self.training_args.dataset.startswith('combine'):
                if raw_datasets['train']['dataset_type'][0] == 'spider':
                    dataset_type = 'spider'
                    trainer = SubTrainer(dataset_type, **trainer_args)
                else:
                    raise NotImplementedError
            elif self.training_args.dataset.startswith('spider'):
                dataset_type = 'spider'
                trainer = SubTrainer(dataset_type, **trainer_args)
            else:
                raise NotImplementedError

            max_length = (
                self.training_args.generation_max_length
                if self.training_args.generation_max_length is not None
                else self.data_args.val_max_target_length
            )
            num_beams = self.data_args.num_beams if self.data_args.num_beams is not None else self.training_args.generation_num_beams

            self.logger.info("*** Predict ***")
            output, spider_results = trainer.predict(
                predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
            )

            metrics = output.metrics
            max_predict_samples = (
                self.data_args.max_predict_samples if self.data_args.max_predict_samples is not None else len(
                    predict_dataset)
            )
            metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
            predictions = tokenizer.batch_decode(
                output.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]

            # save metrics & predict results
            output_dir = f'{self.args.output_dir}/metrics/task_{cur_task_id}/task_{to_eval_task_id}'
            output_eval_result_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_result_file, 'w', encoding='utf-8') as writer:
                writer.write("\n".join(spider_results))
            output_prediction_file = os.path.join(output_dir, "generated_predictions.txt")
            with open(output_prediction_file, "w", encoding='utf-8') as writer:
                writer.write("\n".join(predictions))


        output_dir = f'{self.args.output_dir}/metrics'
        os.makedirs(output_dir, exist_ok=True)

        for cur_task_id in range(self.task_num):
            eval_range = range(cur_task_id + 1) if cur_task_id == self.task_num - 1 else range(cur_task_id + 2)
            for to_eval_task_id in eval_range:
                os.makedirs(f'{output_dir}/task_{cur_task_id}/task_{to_eval_task_id}', exist_ok=True)
                if to_eval_task_id == cur_task_id:
                    source_dir = f'{self.args.output_dir}/task_{cur_task_id}/'
                    target_dir = f'{output_dir}/task_{cur_task_id}/task_{to_eval_task_id}'
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.copy(f'{source_dir}/eval_results.txt', f'{target_dir}/eval_results.txt')
                    shutil.copy(f'{source_dir}/generated_predictions.txt', f'{target_dir}/generated_predictions.txt')
                else:
                    predict(cur_task_id, to_eval_task_id)

        results = {}

        data_dir = self.args.dataset_dir

        for task_id in range(self.task_num):
            results[f'task_{task_id}'] = {}
            results[f'task_{task_id}']['acc_a'], results[f'task_{task_id}']['acc_w'] = \
                compute_acc(self.args.output_dir, task_id, data_dir, self.task_num)
        for task_id in range(1, self.task_num):
            results[f'task_{task_id}']['bwt'] = compute_bwt(self.args.output_dir, task_id)

        for task_id in range(self.task_num - 1):
            results[f'task_{task_id}']['fwt'] = compute_fwt(self.args.output_dir, task_id)
