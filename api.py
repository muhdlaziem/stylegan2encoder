import argparse

from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer
import numpy as np
from configparser import ConfigParser
from werkzeug.utils import secure_filename

import requests
from io import BytesIO
import os
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import PIL.Image
import pretrained_networks
from encoder.generator_model import Generator
import base64
import time
import uuid
import json
import pika

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

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

def load_model():
    print('Loading Generator...')
    _G, _D, Gs = pretrained_networks.load_networks('gdrive:networks/stylegan2-ffhq-config-f.pkl')
    generator = Generator(Gs, batch_size=1, randomize_noise=False)

    return generator

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
    generator = load_model()
    fatness_direction = np.load('directions/fatness_direction.npy')
    print('Models Loaded...')

    UPLOAD_FOLDER = config['server'].get('upload_folder')

    app = Flask(__name__)
    app.secret_key = "secret key"


    @app.route('/projection', methods=['POST'])
    def projection():
        start_time = time.time()

        message = request.get_json(force=True)
        encoded = message['image']
        file_id = str(uuid.uuid4())
        req = json.dumps({
            'method': 'projection',
            'image' : encoded,
            'id' : file_id
        })
        
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        channel = connection.channel()
        channel.queue_declare(queue='', exclusive=True)
        channel.basic_publish(
            exchange='',
            routing_key=config['server'].get('queue_name'),
            body=req,
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
                correlation_id=file_id,
            ))
        connection.close()
        return jsonify({
            'status': 'OK',
            'id': file_id,
        })

    @app.route('/transform', methods=['POST'])
    def transform():
        message = request.get_json(force=True)
        id = message['id']
        coeff = float(message['coeff'])
        path = os.path.join('UPLOAD_FOLDER',f'{id}.npy')

        try:

            latent = np.load(path)
            print(f"Generating images for {id}....")
            original_image = image_to_base64(PIL.Image.open(os.path.join(UPLOAD_FOLDER,f'{id}.png')))
            transformed_image = move_and_show(latent, fatness_direction, coeff, generator)
            print(f"Done Generating images for {id}....")
            return jsonify({
                'status':'OK',
                'original_image': original_image,
                'transformed_image' : image_to_base64(transformed_image)
            })
        except Exception as e:
            return jsonify({
                'status':'error',
                'error': str(e)
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
