import argparse

from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from gevent.pywsgi import WSGIServer
import numpy as np
from configparser import ConfigParser
from werkzeug.utils import secure_filename

import requests
from io import BytesIO
import os
import sys
from ffhq_dataset.face_alignment import image_align
import shutil
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import dataset_tool
from training import dataset
from training import misc
import PIL.Image
import pickle
import hashlib
import bz2
from keras.utils import get_file
import pretrained_networks
import projector
from encoder.generator_model import Generator
from ffhq_dataset.landmarks_detector import LandmarksDetector
import base64
import time

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def project_image(proj, src_file, tmp_dir='.stylegan2-tmp', video=False):

    data_dir = '%s/dataset' % tmp_dir
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    image_dir = '%s/images' % data_dir
    tfrecord_dir = '%s/tfrecords' % data_dir
    os.makedirs(image_dir, exist_ok=True)
    # shutil.copy(src_file, image_dir + '/')
    src_file.save(os.path.join(image_dir, 'img.png'))
    dataset_tool.create_from_images(tfrecord_dir, image_dir, shuffle=0)
    dataset_obj = dataset.load_dataset(
        data_dir=data_dir, tfrecord_dir='tfrecords',
        max_label_size=0, repeat=False, shuffle_mb=0
    )

    # print('Projecting image "%s"...' % os.path.basename(src_file))
    images, _labels = dataset_obj.get_minibatch_np(1)
    images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
    proj.start(images)
    if video:
        video_dir = '%s/video' % tmp_dir
        os.makedirs(video_dir, exist_ok=True)
    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
        if video:
            filename = '%s/%08d.png' % (video_dir, proj.get_cur_step())
            misc.save_image_grid(proj.get_images(), filename, drange=[-1,1])
    print('\r%-30s\r' % '', end='', flush=True)

    # os.makedirs(dst_dir, exist_ok=True)
    # filename = os.path.join(dst_dir, os.path.basename(src_file)[:-4] + '.png')
    # misc.save_image_grid(proj.get_images(), filename, drange=[-1,1])
    # filename = os.path.join(dst_dir, os.path.basename(src_file)[:-4] + '.npy')
    # np.save(filename, proj.get_dlatents()[0])
    shutil.rmtree(tmp_dir)
    return proj.get_dlatents()[0]

def align_images(image_path, landmarks_detector):
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """
    # RAW_IMAGES_DIR = sys.argv[1]
    # ALIGNED_IMAGES_DIR = sys.argv[2]

    imgs = []
    for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(image_path), start=1):
        # face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
        # aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
        # os.makedirs(ALIGNED_IMAGES_DIR, exist_ok=True)
        imgs.append(image_align(image_path, face_landmarks))

    return imgs

def generate_image(latent_vector, generator):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img

def move_and_show(latent_vector, direction, coeff, generator):
    new_latent_vector = latent_vector.copy()
    new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
    return generate_image(new_latent_vector, generator)

def unpack_bz2(src_path):
  data = bz2.BZ2File(src_path).read()
  dst_path = src_path[:-4]
  with open(dst_path, 'wb') as fp:
      fp.write(data)
  return dst_path

def load_model():
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

    return  proj, generator, landmarks_detector

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(img):
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str.decode()

def main(args):

    config = ConfigParser()
    config.read(args.config)

    print('Loading Models...')
    proj, generator, landmarks_detector = load_model()
    fatness_direction = np.load('directions/fatness_direction.npy')
    print('Models Loaded...')

    UPLOAD_FOLDER = 'static/uploads/'

    app = Flask(__name__)
    app.secret_key = "secret key"
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


    # @app.route('/')
    # def upload_form():
    #     return render_template('home.html')

    @app.route('/projection', methods=['POST'])
    def projection():
        start_time = time.time()

        message = request.get_json(force=True)
        encoded = message['image']
        decoded = base64.b64decode(encoded)

        hashed = hashlib.md5(decoded).hexdigest()
        filename = f'{hashed}'

        path = os.path.join(app.config['UPLOAD_FOLDER'],f'{filename}.png')
        print(f'Hash Value: {hashed}, {path}')
        
        img = PIL.Image.open(BytesIO(decoded))
        img.save(path)
        print('Aligning Your Image')
        im = align_images(path, landmarks_detector)
        print('Projecting your image to latent space')
        latent = project_image(proj, im[0], tmp_dir=hashed)
        path_npy = os.path.join(app.config['UPLOAD_FOLDER'],f'{filename}.npy')
        np.save(path_npy, latent)
        print('Image successfully encoded and displayed below')

        return jsonify({
            'status': 'OK',
            'id': hashed,
            'timetaken': '%.1f s' % (time.time() - start_time)
        })

    @app.route('/transform', methods=['POST'])
    def transform():
        message = request.get_json(force=True)
        id = message['id']
        coeff = float(message['coeff'])
        path = os.path.join(app.config['UPLOAD_FOLDER'],f'{id}.npy')
        latent = np.load(path)
        print(f"Generating images for {id}....")
        original_image = generate_image(latent, generator)
        transformed_image = move_and_show(latent, fatness_direction, coeff, generator)
        print(f"Done Generating images for {id}....")
        return jsonify({
            'status':'OK',
            'original_image': image_to_base64(original_image),
            'transformed_image' : image_to_base64(transformed_image)
        })

    host = config['server'].get('host')
    port = int(config['server'].get('port'))

    print('Starting server on host %s port %d' % (host, port))

    http_server = WSGIServer((host, port), app)
    http_server.serve_forever()        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='settings.conf', type=str)

    main(parser.parse_args())
