import collections
import json


def load_json(json_file) -> list:
    with open(json_file, 'r', encoding='utf-8') as f:
        ex_list = json.load(f)
    return ex_list


def get_db_schema(dataset_dir) -> dict:
    tables_json = load_json(dataset_dir + '/tables.json')
    db_schema = collections.defaultdict(dict)
    for table_json in tables_json:
        db_id = table_json['db_id']
        db_schema[db_id] = collections.defaultdict(dict)

        table_id_to_column_ids = collections.defaultdict(list)
        column_id_to_column_name = {}
        column_id_to_table_id = {}
        for column_id, table_column in enumerate(table_json['column_names_original']):
            table_id = table_column[0]
            column_name = table_column[1].replace(' ', '_')
            column_id_to_column_name[column_id] = column_name
            table_id_to_column_ids[table_id].append(column_id)
            column_id_to_table_id[column_id] = table_id

        column_id_to_column_type = {}
        for column_id, column_type in enumerate(table_json['column_types']):
            column_id_to_column_type[column_id] = column_type

        table_id_to_table_name = {}
        table_names = table_json['table_names_original'] if len(table_json['table_names_original']) > 0 else \
            table_json['table_names']
        for table_id, table_name in enumerate(table_names):
            table_id_to_table_name[table_id] = table_name.replace(' ', '_')
        primary_keys = table_json['primary_keys']
        foreign_keys = {}
        for column_id, referenced_column_id in table_json['foreign_keys']:
            foreign_keys[column_id] = referenced_column_id

        for table_id in table_id_to_table_name.keys():
            table_name = table_id_to_table_name[table_id]
            for column_id in table_id_to_column_ids[table_id]:
                column_name = column_id_to_column_name[column_id]

                db_schema[db_id][table_name][column_name] = None

    return db_schema


def generate_template(db_id: str, db_schema: dict):
    tables = []
    for table in db_schema[db_id]:
        table_info = f'{table}: '
        columns = []
        for column in db_schema[db_id][table].keys():
            column_info = column
            columns.append(column_info)
        table_info += ', '.join(columns)
        tables.append(table_info)

    return '; '.join(tables)


def wrap_question(question: str, db_id: str, templates: dict):
    return ' | '.join([templates[db_id], question])


if __name__ == '__main__':
    db_schema = get_db_schema('datasets/spider')

    templates = {}
    for db_id in db_schema.keys():
        templates[db_id] = generate_template(db_id, db_schema)

    question = 'Find the names of customers who either have an deputy policy or uniformed policy.'
    db_id = 'insurance_and_eClaims'

    print(wrap_question(question, db_id, templates))

