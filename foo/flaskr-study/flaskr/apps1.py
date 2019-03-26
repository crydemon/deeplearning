from flask import Flask

app = Flask(__name__, static_url_path='', root_path='', static_folder='C:/Users/wqkant/PycharmProjects/deeplearning/foo/flaskr-study/static')


@app.route('/')
def hello_world():
    return app.send_static_file('html/d3_1.html')


if __name__ == '__main__':
    print(app.static_folder)
    app.run(debug=True)
