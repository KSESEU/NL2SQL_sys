import sqlite3
from args import init_arg_parser
import os
import sys
sys.path.append("..")
from predictor import Predictor
from preprocess_dataset import translate_cn_sql
import json


class Server:
    def __init__(self):
        self.args = init_arg_parser()
        self.nl2sql = Predictor(self.args.model_path, self.args.prop_data_path.format(self.args.data_name))
        self.db_path = self.args.data_path.format(self.args.data_name)
        self.db_list = os.listdir(self.db_path)
        self.db_table_dict = self.get_all_db_table()
        self.direct_dict = self.get_direct_dict() if self.args.use_direct and self.args.data_name == "esql" and os.path.exists(self.args.sample_path.format(self.args.data_name)) else {}


    def sql_excute(self, db_name, sql):
        print("sql: {}".format(sql))
        conn = sqlite3.connect(os.path.join(self.db_path, db_name, "{}.sqlite".format(db_name)))
        c = conn.cursor()
        try:
            result = c.execute(sql)
            return result.fetchall()
        except:
            return []

    def get_sql(self, db_name, nlq):
        sql = self.nl2sql.predict(question=nlq, db_id=db_name)
        if nlq in self.direct_dict:
            sql = self.direct_dict[nlq]
        if "esql" in self.args.data_name and self.args.use_cn_translate:
            sql = translate_cn_sql(sql)
        return sql

    def get_direct_dict(self):
        return {json.loads(x)["query"]: json.loads(x)["sql"] for x in open(self.args.sample_path.format(self.args.data_name), "r", encoding="utf-8")}

    def load_database(self):
        return self.db_list

    def load_table(self, db_name):
        conn = sqlite3.connect(os.path.join(self.db_path, db_name, "{}.sqlite".format(db_name)))
        c = conn.cursor()
        c.execute("select name from sqlite_master where type='table'")
        table_name_list = c.fetchall()
        table_name_list = [line[0] for line in table_name_list]
        return table_name_list

    def get_all_db_table(self):
        db_table_dict = {}
        for db in self.db_list:
            db_table_dict[db] = self.load_table(db)
        return db_table_dict

    def show_table(self, db_name, table_name):
        if table_name not in self.db_table_dict[db_name]:
            return [], []
        conn = sqlite3.connect(os.path.join(self.db_path, db_name, "{}.sqlite".format(db_name)))
        c = conn.cursor()
        try:
            c.execute('pragma table_info({})'.format(table_name))
            col_name_list = c.fetchall()
            col_name_list = [x[1] for x in col_name_list]
            print(col_name_list)

            c.execute('select * from {}'.format(table_name))
            value_list = c.fetchall()
            value_list = [list(x) for x in value_list][:min(100, len(value_list))]
            print(value_list)
        except:
            col_name_list, value_list = [], []
        return col_name_list, value_list




if __name__ == "__main__":
    s = Server()
    s.show_table("singer", "singer")