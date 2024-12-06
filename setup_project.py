import os

# 创建项目结构
base_dir = "audio2text"
directories = [
    "",
    "static",
    "templates",
    "uploads"
]

# 创建目录
for dir in directories:
    path = os.path.join(base_dir, dir)
    os.makedirs(path, exist_ok=True)

# 创建配置文件
config_content = '''# 百度语音识别API配置
APP_ID = "你的APP_ID"
API_KEY = "你的API_KEY"
SECRET_KEY = "你的SECRET_KEY"
'''

# 创建requirements.txt
requirements_content = '''flask
baidu-aip
pydub
python-dotenv
'''

# 创建主应用文件
app_content = '''from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, Audio to Text!"

if __name__ == '__main__':
    app.run(debug=True)
'''

# 写入文件
files = {
    'config.py': config_content,
    'requirements.txt': requirements_content,
    'app.py': app_content
}

for filename, content in files.items():
    with open(os.path.join(base_dir, filename), 'w', encoding='utf-8') as f:
        f.write(content)

print("项目结构创建完成！")
