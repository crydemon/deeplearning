from flask import Flask
from flask import render_template

static_path = '../static'
app = Flask(__name__, static_url_path='', root_path='', static_folder=static_path, template_folder=static_path)


@app.route('/')
def hello_world():
    dataset = [250, 210, 170, 130, 260]
    return render_template('html/p1.html', dataset=dataset)


if __name__ == '__main__':
    print(app.static_folder)
    app.run(debug=True)
