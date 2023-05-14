from transformers import AutoTokenizer, T5ForConditionalGeneration

from utils.utils import get_db_schema, generate_template, wrap_question

#NL2SQL算法接口，用于预测SQL
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
        print(input_text)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        print(input_ids)
        # outputs = self.model.generate(input_ids, max_length=256, num_beams=4)
        outputs = self.model.generate(input_ids, max_length=256, num_beams=4)
        print(outputs)
        sql = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return sql


if __name__ == '__main__':
    # predictor = Predictor(ckpt_dir='ckpt/finetune_t5-small/spider_perm_1/task_0',
    #                       dataset_dir='datasets/spider')
    #
    # question = 'Find the names of customers who either have an deputy policy or uniformed policy.'
    # db_id = 'insurance_and_eClaims'

    predictor = Predictor(ckpt_dir='ckpt/esql_t5-small-chinese_cn/esql/task_0/',
                          dataset_dir='datasets/esql/original_data')

    question = '在制品总价值最低的单位总共有多少'
    sql_origin = "SELECT COUNT(公司名称) FROM 表8-1 WHERE 货品滞销率 > 59.28% AND 货品滞销率 != 18.31%"
    sql_origin_toks = predictor.tokenizer.tokenize(sql_origin)
    print(sql_origin_toks)
    db_id = 'esql_db_7'

    sql = predictor.predict(question=question, db_id=db_id)

    keyword_dict = {"最大": "MAX", "最小": "MIN", "计数": "COUNT", "总和": "SUM", "平均": "AVG",
                    "降序": "DESC", "升序": "ASC", "在之间": "BETWEEN", "与": "AND", "或": "OR",
                    "选择": "SELECT", "关联": "FROM", "条件为": "WHERE", "排序": "ORDER BY", "限制": "LIMIT"}
    sql = sql.replace(" ", "")
    for kw, v in keyword_dict.items():
        print("bbbb")
        if kw in sql:
            print(kw)
            if kw != "与" or (kw == "与" and "与服务费成本" not in sql and "与折让" not in sql):
                print("aaaa")
                sql = sql.replace(kw, " {} ".format(v))
    print(sql)

