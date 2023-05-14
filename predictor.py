from transformers import AutoTokenizer, T5ForConditionalGeneration

from utils.utils import get_db_schema, generate_template, wrap_question


class Predictor:
    def __init__(self, ckpt_dir: str, dataset_dir: str):
        self.templates = {}
        db_schema = get_db_schema(dataset_dir)
        for db_id in db_schema.keys():
            self.templates[db_id] = generate_template(db_id, db_schema)
        self.model = T5ForConditionalGeneration.from_pretrained(ckpt_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)

    def predict(self, question: str, db_id: str):
        input_text = wrap_question(question, db_id, self.templates)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=256, num_beams=4)
        sql = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return sql


if __name__ == '__main__':
    predictor = Predictor(ckpt_dir='ckpt/finetune_t5-small/spider_perm_1/task_0',
                          dataset_dir='datasets/spider')

    question = 'Find the names of customers who either have an deputy policy or uniformed policy.'
    db_id = 'insurance_and_eClaims'

    sql = predictor.predict(question=question, db_id=db_id)

    print(sql)
