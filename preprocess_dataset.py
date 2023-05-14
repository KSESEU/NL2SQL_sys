import json
import pandas as pd
import os

def get_schema(dataset):
    db_list = json.load(open("./datasets/{}/original_data/tables.json".format(dataset), "r", encoding="utf-8"))
    schema_dict = {}
    for db in db_list:
        table_name_list = db["table_names"]
        temp_dict = {name: [] for name in table_name_list}
        for i, col in enumerate(db["column_names"][1:]):
            temp_dict[table_name_list[col[0]]].append([col[1], db["column_types"][i]])
        schema_dict[db["db_id"]] = temp_dict
    return schema_dict


def process_data(data_list, schema_dict):
    new_data_list = []
    for data in data_list:
        db_id, query, query_toks, sql = data
        model_input = ""
        for k, v in schema_dict[db_id].items():
            model_input += "{}: ".format(k)
            model_input += ", ".join([x[0] for x in v])
            model_input += "; "
        model_input = model_input[:-2] + " | " + query
        new_data = {"db_id": db_id, "text": model_input, "sql": sql.lower(), "query": query, "query_toks": query_toks, "example": {"db_id": db_id}}
        new_data_list.append(new_data)
    return new_data_list


def build_db_file_for_esql():
    table_list = [json.loads(line) for line in open("./datasets/esql/original_data/tables.jsonl", "r", encoding="utf-8")]
    db_list = []
    for i, table in enumerate(table_list):
        column_names = [[-1, "*"]] + [[0, name] for name in table["header"]]
        column_types = [x if x == "number" else "text" for x in table["types"]]
        db_id = "esql_db_{}".format(i)

        db = {"column_names": column_names, "column_names_original": column_names, "column_types": column_types, "db_id": db_id,
              "foreign_keys": [], "primary_keys": [], "table_names": ["表{}_1".format(i)], "table_names_original": ["表{}_1".format(i)]}
        db_list.append(db)
    json.dump(db_list, open("./datasets/esql/original_data/tables.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)


def build_dataset_for_esql():
    # agg_list = [None, 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    agg_list = [None, '最大值', '最小值', '计数值', '总和值', '平均值']
    # order_list = ['DESC', 'ASC', None]
    order_list = ['降序', '升序', None]
    # op_list = ['BETWEEN', '=', '>', '<', '>=', '<=', '!=']
    op_list = ['在之间', '=', '>', '<', '>=', '<=', '!=']
    # conn_list = [None, 'AND', 'OR']
    conn_list = [None, '以及', '或者']
    schema_dict = get_schema("esql")
    for name in ["train", "dev", "test"]:
        data_list = [json.loads(line) for line in open("./datasets/esql/original_data/{}.jsonl".format(name), "r", encoding="utf-8")]
        new_data_list = []
        for data in data_list:
            num = int(data["table_id"].split("_")[-1])
            db_id = "esql_db_{}".format(num)
            col_list =  [x[0] for x in schema_dict[db_id]["表{}_1".format(num)]]
            type_list =  [x[1] for x in schema_dict[db_id]["表{}_1".format(num)]]
            query = data["question"]
            query_toks = [x for x in query]
            # sql = "SELECT "
            sql = "选择 "
            #select
            sel_col = [col_list[x] for x in data["sql"]["sel"]]
            for col, agg in zip(sel_col, data["sql"]["agg"]):
                if agg:
                    sql += "{}({}), ".format(agg_list[agg], col)
                else:
                    sql += "{}, ".format(col)
            sql = sql[:-2]

            #from
            # sql += " FROM {} ".format("表{}_1".format(num))
            sql += " 关联 {} ".format("表{}_1".format(num))
            #where
            # sql += "WHERE "
            sql += "条件为 "
            for i, cond in enumerate(data["sql"]["conds"]):
                t = type_list[cond[2]]
                if cond[1] != 0:
                    sql += "{} {} {} ".format(col_list[cond[2]],
                                              op_list[cond[1]],
                                              cond[3] if t == "number" else "'{}'".format(cond[3]))
                    # sql += "{}%{}%{}%".format(col_list[cond[2]],
                    #                           op_list[cond[1]],
                    #                           cond[3] if t == "number" else "'{}'".format(cond[3]))
                else:
                    # sql += "{} {} {} AND {} ".format(col_list[cond[2]],
                    #                                  op_list[cond[1]],
                    #                                  cond[3] if t == "number" else "'{}'".format(cond[3]),
                    #                                  cond[4] if t == "number" else "'{}'".format(cond[4]))
                    sql += "{} {} {} 以及 {} ".format(col_list[cond[2]],
                                                     op_list[cond[1]],
                                                     cond[3] if t == "number" else "'{}'".format(cond[3]),
                                                     cond[4] if t == "number" else "'{}'".format(cond[4]))
                if i < len(data["sql"]["conds"]) - 1:
                    sql += "{} ".format(conn_list[data["sql"]["cond_conn_op"]])
                    # sql += "{}%".format(conn_list[data["sql"]["cond_conn_op"]])
            #order by
            if data["sql"]["ord_by"][0] != -1:
                ob = data["sql"]["ord_by"]
                # sql += "ORDER BY '{}' {} LIMIT {}".format(col_list[ob[0]], order_list[ob[1]], ob[2])
                sql += "排序 {} {} 限制 {}".format(col_list[ob[0]], order_list[ob[1]], ob[2])
            else:
                sql = sql[:-1]
            print(sql)
            new_data_list.append([db_id, query, query_toks, sql])
        processed_data_list = process_data(new_data_list, schema_dict)
        with open("./datasets/esql/original_data/{}_seq2seq.jsonl".format(name), "w", encoding="utf-8") as f:
            for data in processed_data_list:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")


#翻译ESQL的定制化中文SQL（T5-small-chinese对英文特别敏感，极容易产生n-gram repeat）
def translate_cn_sql(sql):
    keyword_dict = {"最大值": "MAX", "最小值": "MIN", "计数值": "COUNT", "总和值": "SUM", "平均值": "AVG",
                    "降序": "DESC", "升序": "ASC", "在之间": "BETWEEN", "以及": "AND", "或者": "OR",
                    "选择": "SELECT", "关联": "FROM", "条件为": "WHERE", "排序": "ORDER BY", "限制": "LIMIT"}
    sql = sql.replace(" ", "").replace("&", "").replace("where", " WHERE ").replace("表1-1", "表1_1")
    for kw, v in keyword_dict.items():
        if kw in sql:
            sql = sql.replace(kw, " {} ".format(v))
    return sql



def process_excel_data(excel_path, excel_name, dataset):
    schema_dict = get_schema(dataset)
    for name in ["train", "dev", "test"]:
        sheet = pd.read_excel(os.path.join(excel_path, excel_name), sheet_name=name, engine="openpyxl")
        data_list = []
        for data in sheet.values:
            print(data)
            data_list.append([data[0], data[1], [x for x in data[1]], data[2]])
        new_data_list = process_data(data_list, schema_dict)
        with open(os.path.join(excel_path, "{}.json".format(name)), "w", encoding="utf-8") as f:
            for data in new_data_list:
                f.write(json.dumps(data, ensure_ascii=False, indent=4) + "\n")


if __name__ == "__main__":
    build_db_file_for_esql()
    build_dataset_for_esql()
