import json
import sqlite3
import os

# 用于为ESQL生成数据库
def create_db(table_list, db_path):
    type_dict = {"string": "text", "number": "float", "date": "text"}


    #创建表格
    for i, table in enumerate(table_list):
        path = os.path.join(db_path, "esql_db_{}".format(i))
        if not os.path.exists(path):
            os.makedirs(path)
        db = sqlite3.connect(os.path.join(path, "esql_db_{}.sqlite".format(i)))
        schema = ""
        for col, type in zip(table["header"], table["types"]):
            schema += "'{}' {}, ".format(col, type_dict[type])
        sql = "create table {}({})".format(table["name"], schema[:-2])
        print(sql)
        db.execute(sql)


    for i, table in enumerate(table_list):
        path = os.path.join(db_path, "esql_db_{}".format(i))
        db = sqlite3.connect(os.path.join(path, "esql_db_{}.sqlite".format(i)))
        c = db.cursor()
        schema = ",".join(["'{}'".format(x) for x in table["header"]])
        for value in table["rows"]:
            v_list = []
            for v, type in zip(value, table["types"]):

                if type == "number":
                    if "%" in v:
                        v = float(v[:-1]) / 100
                    v_list.append(str(float(v)))
                else:
                    v_list.append("'{}'".format(str(v)))

            sql = "insert into {} ({}) values ({})".format(table["name"], schema, ",".join(v_list))
            print(sql)
            c.execute(sql)



if __name__ == "__main__":
    table_path = "datasets/esql/original_data/tables.jsonl"
    db_path = "database/esql/"
    table_list = [json.loads(line) for line in open(table_path, "r", encoding="utf-8")]
    create_db(table_list, db_path)
