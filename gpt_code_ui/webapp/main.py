# The GPT web UI as a template based Flask app
import os
import requests
import asyncio
import json
import re
import logging
import sys
import openai
import pandas as pd

from collections import deque

from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory, Response
from dotenv import load_dotenv

from kernel_program.main import APP_PORT as KERNEL_APP_PORT

load_dotenv('.env')

openai.api_version = os.environ.get("OPENAI_API_VERSION")
openai.log = os.getenv("OPENAI_API_LOGLEVEL")
OPENAI_EXTRA_HEADERS = json.loads(os.environ.get("OPENAI_EXTRA_HEADERS", "{}"))

if openai.api_type == "open_ai":
    AVAILABLE_MODELS = json.loads(os.environ.get("OPENAI_MODELS", '''[{"displayName": "GPT-3.5", "name": "gpt-3.5-turbo"}, {"displayName": "GPT-4", "name": "gpt-4"}]'''))
elif openai.api_type == "azure":
    try:
        AVAILABLE_MODELS = json.loads(os.environ["AZURE_OPENAI_DEPLOYMENTS"])
    except KeyError as e:
        raise RuntimeError('AZURE_OPENAI_DEPLOYMENTS environment variable not set') from e
else:
    raise ValueError(f'Invalid OPENAI_API_TYPE: {openai.api_type}')

UPLOAD_FOLDER = 'workspace/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


APP_PORT = int(os.environ.get("WEB_PORT", 8080))


class LimitedLengthString:
    def __init__(self, maxlen=2000):
        self.data = deque()
        self.len = 0
        self.maxlen = maxlen

    def append(self, string):
        self.data.append(string)
        self.len += len(string)
        while self.len > self.maxlen:
            popped = self.data.popleft()
            self.len -= len(popped)

    def get_string(self):
        result = ''.join(self.data)
        return result[-self.maxlen:]


message_buffer = LimitedLengthString()


def allowed_file(filename):
    return True


def inspect_file(filename: str) -> str:
    READER_MAP = {
        '.csv': pd.read_csv,
        '.tsv': pd.read_csv,
        '.xlsx': pd.read_excel,
        '.xls': pd.read_excel,
        '.xml': pd.read_xml,
        '.json': pd.read_json,
        '.hdf': pd.read_hdf,
        '.hdf5': pd.read_hdf,
        '.feather': pd.read_feather,
        '.parquet': pd.read_parquet,
        '.pkl': pd.read_pickle,
        '.sql': pd.read_sql,
    }

    _, ext = os.path.splitext(filename)

    try:
        df = READER_MAP[ext.lower()](filename)
        return f'这个文件有以下列: {", ".join(df.columns)}'
    except KeyError:
        return ''  # unsupported file type
    except Exception:
        return ''  # file reading failed. - Don't want to know why.


async def get_code(user_prompt, user_openai_key=None, model="gpt-3.5-turbo"):

    prompt = f"""首先，这是我之前要求你完成的任务的历史记录： 
    实际提示如下：
    {message_buffer.get_string()}
    请使用三个反引号的Markdown代码块编写以下Python代码，完成以下任务：
    {user_prompt}

    请注意：

    -  在处理之前，请逐步思考要做什么并用英语写下来。
    -  然后，在代码块中生成有效的Python代码。
    -  确保所有代码有效，可以在Jupyter Python 3内核环境中运行。
    -  在使用变量之前，请先定义它。
    -  对于数据转换，您可以使用以下库：
        - 'numpy'  # numpy==1.24.3
        - 'dateparser'  # dateparser==1.1.8
        - 'pandas'  # pandas==1.5.3
        - 'geopandas'  # geopandas==0.13.2
    -  对于PDF提取，您可以使用以下库：
        - 'PyPDF2'  # PyPDF2==3.0.1
        - 'pdfminer'  # pdfminer==20191125
        - 'pdfplumber'  # pdfplumber==0.9.0
    -  对于数据可视化，您可以使用以下库：
        - 'matplotlib'  # matplotlib==3.7.1

    请确保使用matplotlib生成图表。如果需要地理图表，请使用geopandas和geopandas.datasets模块。

    如果用户刚刚上传了一个文件，请关注最近上传的文件（以及可选地关注之前上传的所有文件）。

    如果代码修改或生成了一个文件，在代码块的末尾插入一个打印语句，以将其作为HTML字符串打印为链接：<a href='/download?file=INSERT_FILENAME_HERE'>Download file</a>。将INSERT_FILENAME_HERE替换为实际的文件名。"""

    if user_openai_key:
        openai.api_key = user_openai_key

    arguments = dict(
        temperature=0.7,
        headers=OPENAI_EXTRA_HEADERS,
        messages=[
            # {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    )

    if openai.api_type == 'open_ai':
        arguments["model"] = model
    elif openai.api_type == 'azure':
        arguments["deployment_id"] = model
    else:
        return None, f"Error: Invalid OPENAI_PROVIDER: {openai.api_type}", 500

    try:
        result_GPT = openai.ChatCompletion.create(**arguments)

        if 'error' in result_GPT:
            raise openai.APIError(code=result_GPT.error.code, message=result_GPT.error.message)

        if result_GPT.choices[0].finish_reason == 'content_filter':
            raise openai.APIError('Content Filter')

    except openai.OpenAIError as e:
        return None, f"Error from API: {e}", 500

    try:
        content = result_GPT.choices[0].message.content

    except AttributeError:
        return None, f"Malformed answer from API: {content}", 500

    def extract_code(text):
        # Match triple backtick blocks first
        triple_match = re.search(r'```(?:\w+\n)?(.+?)```', text, re.DOTALL)
        if triple_match:
            return triple_match.group(1).strip()
        else:
            # If no triple backtick blocks, match single backtick blocks
            single_match = re.search(r'`(.+?)`', text, re.DOTALL)
            if single_match:
                return single_match.group(1).strip()

    return extract_code(content), content.strip(), 200

# We know this Flask app is for local use. So we can disable the verbose Werkzeug logger
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

cli = sys.modules['flask.cli']
cli.show_server_banner = lambda *x: None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CORS(app)


@app.route('/')
def index():

    # Check if index.html exists in the static folder
    if not os.path.exists(os.path.join(app.root_path, 'static/index.html')):
        print("index.html not found in static folder. Exiting. Did you forget to run `make compile_frontend` before installing the local package?")

    return send_from_directory('static', 'index.html')


@app.route("/models")
def models():
    return jsonify(AVAILABLE_MODELS)


@app.route('/api/<path:path>', methods=["GET", "POST"])
def proxy_kernel_manager(path):
    if request.method == "POST":
        resp = requests.post(
            f'http://localhost:{KERNEL_APP_PORT}/{path}', json=request.get_json())
    else:
        resp = requests.get(f'http://localhost:{KERNEL_APP_PORT}/{path}')

    excluded_headers = ['content-encoding',
                        'content-length', 'transfer-encoding', 'connection']
    headers = [(name, value) for (name, value) in resp.raw.headers.items()
               if name.lower() not in excluded_headers]

    response = Response(resp.content, resp.status_code, headers)
    return response


@app.route('/assets/<path:path>')
def serve_static(path):
    return send_from_directory('static/assets/', path)


@app.route('/download')
def download_file():

    # Get query argument file
    file = request.args.get('file')
    # from `workspace/` send the file
    # make sure to set required headers to make it download the file
    return send_from_directory(os.path.join(os.getcwd(), 'workspace'), file, as_attachment=True)


@app.route('/inject-context', methods=['POST'])
def inject_context():
    user_prompt = request.json.get('prompt', '')

    # Append all messages to the message buffer for later use
    message_buffer.append(user_prompt + "\n\n")

    return jsonify({"result": "success"})


@app.route('/generate', methods=['POST'])
def generate_code():
    user_prompt = request.json.get('prompt', '')
    user_openai_key = request.json.get('openAIKey', None)
    model = request.json.get('model', None)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    code, text, status = loop.run_until_complete(
        get_code(user_prompt, user_openai_key, model))
    loop.close()

    # Append all messages to the message buffer for later use
    message_buffer.append(user_prompt + "\n\n")

    return jsonify({'code': code, 'text': text}), status


@app.route('/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        file_target = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_target)
        file_info = inspect_file(file_target)
        return jsonify({'message': f'文件 {file.filename} 上传成功.\n{file_info}'}), 200
    else:
        return jsonify({'error': '文件类型错误'}), 400


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=APP_PORT, debug=True, use_reloader=False)
