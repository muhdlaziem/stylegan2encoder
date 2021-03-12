import argparse
import numpy as np
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
import logging
import pika
import json
from configparser import ConfigParser

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
# UPLOAD_FOLDER = 'static/uploads/'

def project_image(proj, src_file, filename, tmp_dir='.stylegan2-tmp', video=False):

    data_dir = '%s/dataset' % tmp_dir
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    image_dir = '%s/images' % data_dir
    tfrecord_dir = '%s/tfrecords' % data_dir
    os.makedirs(image_dir, exist_ok=True)
    src_file.save(os.path.join(image_dir, 'img.png'))
    dataset_tool.create_from_images(tfrecord_dir, image_dir, shuffle=0)
    dataset_obj = dataset.load_dataset(
        data_dir=data_dir, tfrecord_dir='tfrecords',
        max_label_size=0, repeat=False, shuffle_mb=0
    )

    images, _labels = dataset_obj.get_minibatch_np(1)
    images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
    proj.start(images)
    if video:
        video_dir = '%s/video' % tmp_dir
        os.makedirs(video_dir, exist_ok=True)
    while proj.get_cur_step() < proj.num_steps:
        logging.info('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
        if video:
            filename = '%s/%08d.png' % (video_dir, proj.get_cur_step())
            misc.save_image_grid(proj.get_images(), filename, drange=[-1,1])
    logging.info('\r%-30s\r' % '', end='', flush=True)

   
    misc.save_image_grid(proj.get_images(), filename, drange=[-1,1])
    
    shutil.rmtree(tmp_dir)
    return proj.get_dlatents()[0]

def align_images(image_path, landmarks_detector):
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """
    imgs = []
    for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(image_path), start=1):
       
        imgs.append(image_align(image_path, face_landmarks))

    return imgs

def unpack_bz2(src_path):
  data = bz2.BZ2File(src_path).read()
  dst_path = src_path[:-4]
  with open(dst_path, 'wb') as fp:
      fp.write(data)
  return dst_path

def load_model():
    logging.info('Loading Generator...')
    _G, _D, Gs = pretrained_networks.load_networks('gdrive:networks/stylegan2-ffhq-config-f.pkl')
    proj = projector.Projector(
        vgg16_pkl             = 'https://drive.google.com/uc?id=1hPF2dybG3z-s5OYpyiWjePUayutYkpRO',
        num_steps             = 1000,
        initial_learning_rate = 0.1,
        initial_noise_factor  = 0.05,
        verbose               = False
    )
    proj.set_network(Gs)

    logging.info('Loading Landmarks Detector...')
    LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                                LANDMARKS_MODEL_URL, cache_subdir='temp'))
    landmarks_detector = LandmarksDetector(landmarks_model_path)

    return  proj, landmarks_detector


class FatToThinRpc:
    def __init__(self, config):
        super().__init__()
        logging.info('Loading Models...')
        self.proj, self.landmarks_detector = load_model()
        self.UPLOAD_FOLDER = config['server'].get('upload_folder')
        logging.info('Models Loaded...')

    def on_rx_rpc_request(self, channel, method_frame, properties, body):
        logging.debug('RPC Server processing request: %s', body)

        res_props = pika.BasicProperties(correlation_id=properties.correlation_id)

        try:
            req = json.loads(body)
            res = json.dumps({'status': 'empty'})

            if req['method'] == 'projection':
                logging.info('Processing projection RPC for %s', req['id'])
                encoded = req['image']
                decoded = base64.b64decode(encoded)

                filename = req['id']

                path = os.path.join(self.UPLOAD_FOLDER,f'{filename}.png')
                logging.info(f'Image Path: {path}')
                
                img = PIL.Image.open(BytesIO(decoded))
                img.save(path)

                logging.info('Aligning Image')
                im = align_images(path, self.landmarks_detector)

                logging.info('Projecting image to latent space')
                latent = project_image(self.proj, im[0], path, tmp_dir=req['id'])
                path_npy = os.path.join(self.UPLOAD_FOLDER,f'{filename}.npy')
                np.save(path_npy, latent)

                logging.info('Image successfully encoded for %s', req['id'])

                res = json.dumps({
                    'status' : 'OK'
                })
            else:
                logging.warning("Received unknown method: %s", req['method'])
        except Exception as e :
            logging.warning('RPC Server failed to process request %s', str(e))

            res = json.dumps({
                'status' : 'Problem Occured',
                'error': str(e)
            })

        # logging.info('Publishing response: %s', res)

        # channel.basic_publish(
        #     exchange='',
        #     properties=res_props,
        #     routing_key=properties.reply_to,
        #     body=res
        # )
        # channel.basic_ack(delivery_tag=method_frame.delivery_tag)

        logging.info('RPC Server request finished')



def main(args):
    logging.getLogger().setLevel('INFO')
    logging.info('RPC Server starting...')

    config = ConfigParser()
    config.read(args.config)

    try:
        server = FatToThinRpc(config)

        with pika.BlockingConnection() as conn:
            channel = conn.channel()

            channel.queue_declare(
                queue=config['server'].get('queue_name'), 
                exclusive=True, 
                auto_delete=True
            )
            channel.basic_consume(
                config['server'].get('queue_name'),
                server.on_rx_rpc_request
            )

            logging.info('RPC Server ready.')
            channel.start_consuming()
            
    except Exception as e:
        logging.warning('Caught error:')
        logging.warning(e)

    logging.info('RPC Server shutting down')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='settings.conf', type=str)

    main(parser.parse_args())
