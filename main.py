import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

from PIL import  Image
from io import  BytesIO
import base64
import numpy as np
from utils import align_images, move_and_show, project_image, generate_image
import hashlib
import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks
import projector
from encoder.generator_model import Generator
import bz2
from ffhq_dataset.landmarks_detector import LandmarksDetector

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def unpack_bz2(src_path):
  data = bz2.BZ2File(src_path).read()
  dst_path = src_path[:-4]
  with open(dst_path, 'wb') as fp:
      fp.write(data)
  return dst_path

def load_model():
    global proj, generator, landmarks_detector
    print('Loading Generator...')
    _G, _D, Gs = pretrained_networks.load_networks('gdrive:networks/stylegan2-ffhq-config-f.pkl')
    proj = projector.Projector(
        vgg16_pkl             = 'https://drive.google.com/uc?id=1hPF2dybG3z-s5OYpyiWjePUayutYkpRO',
        num_steps             = 1000,
        initial_learning_rate = 0.1,
        initial_noise_factor  = 0.05,
        verbose               = False
    )
    proj.set_network(Gs)

    generator = Generator(Gs, batch_size=1, randomize_noise=False)

    print('Loading Landmarks Detector...')
    LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                                LANDMARKS_MODEL_URL, cache_subdir='temp'))
    landmarks_detector = LandmarksDetector(landmarks_model_path)


print('Loading Models...')
load_model()
fatness_direction = np.load('directions/fatness_direction.npy')
print('Models Loaded...')

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(img):
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str.decode()

@app.route('/')
def upload_form():
	return render_template('home.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
        
    if file and allowed_file(file.filename):
        buf = file.read()
        hashed = hashlib.md5(buf).hexdigest()
        filename = os.path.join(app.config['UPLOAD_FOLDER'],f'{hashed}_{secure_filename(file.filename)}')
        print(f'Hash Value: {hashed}, {filename}')
        img = Image.open(BytesIO(buf))
        img.save(filename)
        flash('Aligning Your Image')
        im = align_images(filename, landmarks_detector)
        flash('Projecting your image to latent space')
        latent = project_image(proj, im[0], tmp_dir=hashed)
        flash('Image successfully encoded and displayed below')

        # yield render_template('home.html', image="data:image/png;base64," + )
        transform = move_and_show(latent, fatness_direction, 0.5, generator)
        flash('Image successfully transformed and displayed below')

        return render_template('home.html', result="data:image/png;base64," + image_to_base64(generate_image(latent, generator)), image="data:image/png;base64," + image_to_base64(transform))
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)



if __name__ == "__main__":
    app.run(port=80,debug=True)
