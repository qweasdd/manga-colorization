from flask import Flask, request, jsonify, abort, redirect, url_for, render_template, send_file
from flask_wtf import FlaskForm
from wtforms import StringField, FileField, BooleanField, DecimalField
from wtforms.validators import DataRequired
from flask import after_this_request

import torch

import os
from model.models import Colorizer, Generator
from model.extractor import get_seresnext_extractor
from utils.xdog import XDoGSketcher
from utils.utils import open_json
from denoising.denoiser import FFDNetDenoiser
from datetime import datetime

from inference import colorize_single_image, colorize_images, colorize_cbr

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

generator = Generator()
generator.load_state_dict(torch.load('model/generator.pth'))

extractor = get_seresnext_extractor()
extractor.load_state_dict(torch.load('model/extractor.pth'))

colorizer = Colorizer(generator, extractor)
colorizer = colorizer.eval().to(device)

sketcher = XDoGSketcher()
xdog_config = open_json('configs/xdog_config.json')
for key in xdog_config.keys():
    if key in sketcher.params:
        sketcher.params[key] = xdog_config[key]

denoiser = FFDNetDenoiser(device)
 

app = Flask(__name__)
app.config.update(dict(
    SECRET_KEY="lol kek",
    WTF_CSRF_SECRET_KEY="cheburek"
))

color_args = {'colorizer':colorizer, 'sketcher':sketcher, 'device':device, 'dfm' : True}
    
class SubmitForm(FlaskForm):
    file = FileField(validators=[DataRequired()])
    denoise = BooleanField(default = 'checked')
    denoise_sigma = DecimalField(label = 'Denoise sigma', validators=[DataRequired()], default = 25, places = None)
    autohint = BooleanField(default = None)
    autohint_sigma = DecimalField(label = 'Autohint sigma', validators=[DataRequired()], default= 0.0003, places = None)
    ignore_gray = BooleanField(label = 'Ignore gray autohint', default = None)
    
@app.route('/img/<path>')
def show_image(path):
    return f'<img src="/static/{path}">'
    
@app.route('/', methods=('GET', 'POST'))
def submit_data():
    form = SubmitForm()
    if form.validate_on_submit():

        input_data = form.file.data
        
        _, ext = os.path.splitext(input_data.filename)
        filename = str(datetime.now()) + ext
        
        input_data.save(filename)
        
        color_args['auto_hint'] = form.autohint.data
        color_args['auto_hint_sigma'] = float(form.autohint_sigma.data)
        color_args['ignore_gray'] = form.ignore_gray.data
        color_args['denoiser'] = None
        
        if form.denoise.data:
            color_args['denoiser'] = denoiser
            color_args['denoiser_sigma'] = float(form.denoise_sigma.data)
        
        if ext.lower() in ('.cbr', '.cbz', '.rar', '.zip'):
            result_name = colorize_cbr(filename, color_args)
            os.remove(filename)
            
            @after_this_request
            def remove_file(response):
                try:
                    os.remove(result_name)
                except Exception as error:
                    app.logger.error("Error removing or closing downloaded file handle", error)
                return response
            
            return send_file(result_name, mimetype='application/vnd.comicbook-rar', attachment_filename=result_name, as_attachment=True)
        
        elif ext.lower() in ('.jpg', '.png', ',jpeg'):
            random_name = str(datetime.now()) + '.png'
            new_image_path = os.path.join('static', random_name)
            
            colorize_single_image(filename, new_image_path, color_args)
            os.remove(filename)
            
            return redirect(f'/img/{random_name}')
        else:
            return 'Wrong format'

    return render_template('submit.html', form=form)