import os
from datetime import datetime
import base64
import random
import string
import shutil
import torch
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, jsonify, abort, redirect, url_for, render_template, send_file, Response
from flask_wtf import FlaskForm
from wtforms import StringField, FileField, BooleanField, DecimalField
from wtforms.validators import DataRequired
from flask import after_this_request

from model.models import Colorizer, Generator
from model.extractor import get_seresnext_extractor
from utils.xdog import XDoGSketcher
from utils.utils import open_json
from denoising.denoiser import FFDNetDenoiser
from inference import process_image_with_hint
from utils.utils import resize_pad
from utils.dataset_utils import get_sketch

def generate_id(size=25, chars=string.ascii_letters + string.digits):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

def generate_unique_id(current_ids = set()):
    id_t = generate_id()
    while id_t in current_ids:
        id_t = generate_id()
        
    current_ids.add(id_t)    
        
    return id_t

app = Flask(__name__)
app.config.update(dict(
    SECRET_KEY="lol kek",
    WTF_CSRF_SECRET_KEY="cheburek"
))

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'  

colorizer = torch.jit.load('./model/colorizer.zip', map_location=torch.device(device))

sketcher = XDoGSketcher()
xdog_config = open_json('configs/xdog_config.json')
for key in xdog_config.keys():
    if key in sketcher.params:
        sketcher.params[key] = xdog_config[key]

denoiser = FFDNetDenoiser(device)

color_args = {'colorizer':colorizer, 'sketcher':sketcher, 'device':device, 'dfm' : True, 'auto_hint' : False, 'ignore_gray' : False, 'denoiser' : denoiser, 'denoiser_sigma' : 25}


class SubmitForm(FlaskForm):
    file = FileField(validators=[DataRequired(), ])

def preprocess_image(file_id, ext): 
    directory_path = os.path.join('static', 'temp_images', file_id)
    original_path = os.path.join(directory_path, 'original') + ext
    original_image = plt.imread(original_path)    
                          
    resized_image, _ = resize_pad(original_image)
    resized_image = denoiser.get_denoised_image(resized_image, 25)
    bw, dfm = get_sketch(resized_image, sketcher, True)
                                
    resized_name = 'resized_' + str(resized_image.shape[0]) + '_' + str(resized_image.shape[1]) + '.png'
    plt.imsave(os.path.join(directory_path, resized_name), resized_image)
    plt.imsave(os.path.join(directory_path, 'bw.png'), bw, cmap = 'gray')
    plt.imsave(os.path.join(directory_path, 'dfm.png'), dfm, cmap = 'gray')
    os.remove(original_path)
  
    empty_hint = np.zeros((resized_image.shape[0], resized_image.shape[1], 4), dtype = np.float32)
    plt.imsave(os.path.join(directory_path, 'hint.png'), empty_hint)

@app.route('/', methods=['GET', 'POST'])
def upload():
    form = SubmitForm()
    if form.validate_on_submit():
        input_data = form.file.data
        
        _, ext = os.path.splitext(input_data.filename)
        
        if ext not in ('.jpg', '.png', '.jpeg'):
            return abort(400)
        
        file_id = generate_unique_id()
        directory = os.path.join('static', 'temp_images', file_id)
        original_filename =  os.path.join(directory, 'original') + ext
        
        try :
            os.mkdir(directory)
            input_data.save(original_filename)
        
            preprocess_image(file_id, ext)
        
            return redirect(f'/draw/{file_id}')
        
        except :
            print('Failed to colorize')
            if os.path.exists(directory):
                shutil.rmtree(directory)
            return abort(400)
                
        
    return render_template("upload.html", form = form)

@app.route('/img/<file_id>')
def show_image(file_id):
    if not os.path.exists(os.path.join('static', 'temp_images', str(file_id))):
        abort(404)
    return f'<img src="/static/temp_images/{file_id}/colorized.png?{random. randint(1,1000000)}">'

def colorize_image(file_id):
    directory_path = os.path.join('static', 'temp_images', file_id)
    
    bw = plt.imread(os.path.join(directory_path, 'bw.png'))[..., :1]
    dfm = plt.imread(os.path.join(directory_path, 'dfm.png'))[..., :1]
    hint = plt.imread(os.path.join(directory_path, 'hint.png'))
    
    return process_image_with_hint(bw, dfm, hint, color_args)

@app.route('/colorize', methods=['POST'])
def colorize():
    
    file_id = request.form['save_file_id']
    file_id = file_id[file_id.rfind('/') + 1:]
    
    img_data = request.form['save_image']
    img_data = img_data[img_data.find(',') + 1:]
    
    directory_path = os.path.join('static', 'temp_images', file_id)
    
    with open(os.path.join(directory_path, 'hint.png'), "wb") as im:
        im.write(base64.decodestring(str.encode(img_data)))
    
    result = colorize_image(file_id)
    
    plt.imsave(os.path.join(directory_path, 'colorized.png'), result)
      
    src_path = f'../static/temp_images/{file_id}/colorized.png?{random. randint(1,1000000)}'    
        
    return src_path

@app.route('/draw/<file_id>', methods=['GET', 'POST'])
def paintapp(file_id):
    if request.method == 'GET':
        
        directory_path = os.path.join('static', 'temp_images', str(file_id))
        if not os.path.exists(directory_path):
            abort(404)
        
        resized_name = [x for x in os.listdir(directory_path) if x.startswith('resized_')][0]
        
        split = os.path.splitext(resized_name)[0].split('_')
        width = int(split[2])
        height = int(split[1])
        
        return render_template("drawing.html", height = height, width = width, img_path = os.path.join('temp_images', str(file_id), resized_name))
