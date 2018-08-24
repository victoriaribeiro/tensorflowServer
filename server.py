
# A very simple Bottle Hello World app for you to get started with...
from bottle import default_app, route, request, static_file, run
import logging, os
from random import randint


@route('/')
def hello_world():
    return 'Hello from Bottle!'

@route('/status/', method='GET')
def status():
    return 'online'

@route('/upload/', method='POST')
def upload():

    upload = request.files.get('image')

    name, ext = os.path.splitext(upload.filename)
    # if ext not in ('.png', '.jpg', '.jpeg'):
    #     return "File extension not allowed."

    save_path = "/home/victoriaribeiro/imagens"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = "{path}/{file}".format(path=save_path, file=upload.filename)
    upload.save(file_path, True)
    return "File successfully saved to '{0}'.".format(file_path)


application = default_app()

