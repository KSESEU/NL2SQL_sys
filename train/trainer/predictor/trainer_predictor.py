import os


class PredictorTrainer:
    def __init__(self, args):
        self.args = args
        self.task_num = self.get_task_num()

    def get_task_num(self):
        """
        get total task num of the continual learning task stream
        :return: task num
        """
        task_num = 0
        for file_or_dir_name in os.listdir(self.args.dataset_dir):
            if file_or_dir_name.startswith('task_'):
                task_num += 1

        return task_num
