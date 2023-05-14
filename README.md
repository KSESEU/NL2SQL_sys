# NL2SQL智能问答系统

## 环境配置
```
transformers=4.21.3
flask
openpyxl
pandas
sqlite3
```

## 文件目录
```

.
├── ckpt                            # 模型文件存放目录
├── database                        # 数据库文件存放目录
├── datasets                        # 数据集文件存放目录
├── utils                           # 存放工具类代码
├── train                           # 训练代码文件夹
├── static                          # 存放css代码以及前端相关素材
├── system                          # 存放系统数据
├── templates                       # 存放html代码
├── predictor.py                    # NL2SQL模型预测接口
├── preprocess_dataset.py           # 预处理数据（被文件上传接口调用，也可直接离线使用）
├── engine.py                       # 用于生成SQL以及查询数据库的代码
├── main.py                         # 服务器代码
├── build_db.py                     # 构建ESQL数据库代码
└── args.py                         # 系统参数设置


```

## 运行系统

1. 首先对main.py文件中的url，按照实际部署的服务器与端口进行修改
```
app.run(port=2021,host="10.201.186.126",debug=True)
```

2. 将模型文件以及查询数据库分别存放于ckpt与database文件夹下的子目录中，并按照实际情况修改args.py中的配置路径

3. 通过一下命令运行系统
```
python main.py
```

## 系统使用

注册/登录功能：首先通过url/login进入登录界面，如果没有注册账号则点击注册链接，并录入用户信息

问答功能：选择账号类型为“用户”进入系统的问答模块，选择 查询目标数据库 以及输入 查询文本，点击确认，后台会调用NL2SQL接口并使用生成SQL查询数据库，将SQL与答案同时显示与界面中

查看数据库功能：通过管理员账号进行登录，即可访问该功能，在页面跳转后，选择数据库点击确认，并选择具体的表格再次点击确认，界面会展示该表格信息

导入训练数据功能：通过点击“跳转至模型管理界面”进入，在左上角选择数据集，并进行结构化EXCEL文件上传，点击确认，后台会自动对EXCEL文件进行解析与预处理，并生成对应的数据集，EXCEL格式如 new_dataset.xlsx 所示

## 系统扩展

界面扩展：在static/css中新增css样式，同时在templates中新增html页面

功能扩展：在main.py中编写新的服务接口
