from flask import Flask, render_template, request, Response, redirect, flash
from engine import Server
import os
import time
import json
from preprocess_dataset import process_excel_data


app = Flask(__name__)
engine = Server()
app.config['SECRET_KEY'] = '123456'


@app.route('/nl2sql', methods=["POST", "GET"])
def nl2sql():
    db_list = engine.db_list
    col_list = []
    value_list = []
    res_sql = ""
    if request.method == "POST":
        db = request.form["db"]
        query = request.form["query"]
        print("Get {} {}".format(db, query))
        sql = engine.get_sql(db, query)
        res_sql = "问题：\n" + query + '\n\n' + "查询数据库：\n\n" + db + '\n' + "生成SQL：\n" + sql
        result = engine.sql_excute(db, sql)
        print(result, list(result), len(list(result)))
        if result:
            sel_str = str(sql).split("where")[0].split("WHERE")[0].split("from")[0].split("FROM")[0].replace("SELECT", "").replace("select", "").strip()
            # col_list = [x.strip() for x in sel_str.split(',')]
            for r in list(result):
                print(r, list(r))
                value_list.append(list(r))

            print(value_list)

    return render_template("nl2sql.html", db_list=db_list, col_list=col_list, value_list=value_list, res_sql=res_sql)


@app.route('/register', methods=["POST", "GET"])
def register():
    if request.method == "POST":
        account = request.form["account"]
        password = request.form["psw"]
        cf_password = request.form["cf_psw"]
        phone_number = request.form["pn"]
        email = request.form["email"]
        account_info = json.load(open("./system/account_user.json", 'r', encoding="utf-8"))
        if password == cf_password and account not in account_info:
            account_info[account] = [password, phone_number, email]
            json.dump(account_info, open("./system/account_user.json", 'w', encoding="utf-8"), ensure_ascii=False, indent=4)
            flash("注册成功！")
            return redirect("/login")
        else:
            flash("注册失败，用户名已存在或密码校验错误！")
            return render_template("register.html")
    else:
        return render_template("register.html")


@app.route('/login', methods=["POST", "GET"])
def login():
    if request.method == "POST":
        account = request.form["account"]
        password = request.form["psw"]
        type = request.form["type"]
        if type == "管理员":
            account_info = json.load(open("./system/account_admin.json", 'r', encoding="utf-8"))
        else:
            account_info = json.load(open("./system/account_user.json", 'r', encoding="utf-8"))
        if account in account_info and account_info[account][0] == password:
            return redirect("./nl2sql" if type == "用户" else "./db_manage")
        else:
            flash("登录失败，请检查用户名，密码与账号类型！")
            return render_template("login.html")
    return render_template("login.html")


@app.route('/db_manage', methods=["POST", "GET"])
def db_manage():
    db_table_dict = engine.db_table_dict
    col_list = []
    value_list = []
    if request.method == "POST":
        db_name = request.form['db']
        table_name = request.form['table']
        col_list, value_list = engine.show_table(db_name, table_name)

    return render_template("db_manage.html", db_table_dict=db_table_dict, col_list=col_list, value_list=value_list)


@app.route('/model_manage')
def model_manage():
    return render_template("model_manage.html")


@app.route('/upload_file', methods=['POST'])
def upload_file():
    print("upload_file")
    if request.method == 'POST':
        path = request.form['dataset']
        if not path:
            flash("请输入数据集名称", category="success")
        else:
            f = request.files['file']
            time_path = time.strftime('%Y%m%d%H%M', time.localtime())
            base_path = os.path.join("datasets", str(path), time_path)
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            data_path = os.path.join(base_path, f.filename)
            f.save(data_path)
            flash("文件上传成功", category="success")
            process_excel_data(base_path, f.filename, path)
    return redirect("./model_manage")



if __name__ == "__main__":
    app.run(port=2020,host="10.201.109.46",debug=True)
    # app.run(port=2020,host="127.0.0.1",debug=True)