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

import threading



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


def image_to_base64(img):
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str.decode()

def save_progress(uuid,mode,status,config):
    path = os.path.join(config['server'].get('upload_folder'), f"{uuid}.json")
    with open(path , "r") as jsonFile:
        data = json.load(jsonFile)
    data[mode] = status

    with open(path, "w") as jsonFile:
        json.dump(data, jsonFile)

def project_image(proj, src_file, filename, tmp_dir='.stylegan2-tmp', video=False, config=None, uuid=None):
    UPLOAD_FOLDER = config['server'].get('upload_folder')

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
        save_progress(uuid,'progress',proj.get_cur_step(), config)

        # save image and latent every 100 steps
        if (int(proj.get_cur_step()) + 1) % 100 == 0 and int(proj.get_cur_step()) != (proj.num_steps - 1) :
            misc.save_image_grid(proj.get_images(), filename, drange=[-1,1])
            path_npy = os.path.join(UPLOAD_FOLDER,f'{uuid}.npy')
            np.save(path_npy, proj.get_dlatents()[0])

        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
        if video:
            filename = '%s/%08d.png' % (video_dir, proj.get_cur_step())
            misc.save_image_grid(proj.get_images(), filename, drange=[-1,1])
    print('\r%-30s\r' % '', end='', flush=True)

   
    misc.save_image_grid(proj.get_images(), filename, drange=[-1,1])

    path_npy = os.path.join(UPLOAD_FOLDER,f'{uuid}.npy')
    np.save(path_npy, proj.get_dlatents()[0])

    shutil.rmtree(tmp_dir)

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


def Projection_model():
    print('Loading Projection_model...')
    _G, _D, Gs = pretrained_networks.load_networks('gdrive:networks/stylegan2-ffhq-config-f.pkl')
    proj = projector.Projector(
        vgg16_pkl             = 'https://drive.google.com/uc?id=1hPF2dybG3z-s5OYpyiWjePUayutYkpRO',
        num_steps             = 1000,
        initial_learning_rate = 0.1,
        initial_noise_factor  = 0.05,
        verbose               = False
    )
    proj.set_network(Gs)

    print('Loading Landmarks Detector...')
    LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                                LANDMARKS_MODEL_URL, cache_subdir='temp'))
    landmarks_detector = LandmarksDetector(landmarks_model_path)

    return  proj, landmarks_detector

def Generator_model():
    print('Loading Generator...')
    _G, _D, Gs = pretrained_networks.load_networks('gdrive:networks/stylegan2-ffhq-config-f.pkl')
    generator = Generator(Gs, batch_size=1, randomize_noise=False)

    return generator

class ProjectionRpc:
    def __init__(self, config):
        super().__init__()
        self.proj, self.landmarks_detector = Projection_model()
        self.UPLOAD_FOLDER = config['server'].get('upload_folder')
        self.config = config

    

    def on_rx_rpc_request(self, channel, method_frame, properties, body):
        logging.debug('RPCProjection Server processing request: %s', body)

        res_props = pika.BasicProperties(correlation_id=properties.correlation_id)

        try:
            req = json.loads(body)
            res = json.dumps({'status': 'empty'})

            if req['method'] == 'projection':
                logging.info('Processing projection RPC for %s', req['id'])
                encoded = req['image']
                decoded = base64.b64decode(encoded)

                filename = req['id']
                with open(os.path.join(self.UPLOAD_FOLDER, f"{filename}.json"), 'w') as fp: 
                    json.dump({'status':'Processing image projection','progress':0}, fp)
                
                path = os.path.join(self.UPLOAD_FOLDER,f'{filename}.png')
                logging.info(f'Image Path: {path}')
                
                img = PIL.Image.open(BytesIO(decoded))
                img.save(path)

                save_progress(filename,'status','Aligning your image', self.config)
                logging.info('Aligning Image')
                im = align_images(path, self.landmarks_detector)

                save_progress(filename,'status','Projecting your image to latent space', self.config)
                logging.info('Projecting image to latent space')
                project_image(self.proj, im[0], path, tmp_dir=req['id'], config=self.config, uuid=filename)
                

                logging.info('Image successfully encoded for %s', req['id'])

                res = json.dumps({
                    'status' : 'OK',
                    'id': req['id']
                })
            else:
                logging.warning("Received unknown method: %s", req['method'])
        except Exception as e :
            logging.warning('RPCProjection Server failed to process request %s', str(e))

            res = json.dumps({
                'status' : 'Problem Occured',
                'error': str(e)
            })

        logging.info('RPCProjection Publishing response: %s', res)

        channel.basic_publish(
            exchange='',
            properties=res_props,
            routing_key=properties.reply_to,
            body=res
        )
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)

        logging.info('RPCProjection Server request finished')


class TransformRpc:
    def __init__(self, config):
        super().__init__()
        self.generator = Generator_model()
        self.UPLOAD_FOLDER = config['server'].get('upload_folder')
        self.fatness_direction = np.load(config['server'].get('fatness_direction'))
        
    def on_rx_rpc_request(self, channel, method_frame, properties, body):
        logging.debug('RPCTransform Server processing request: %s', body)

        res_props = pika.BasicProperties(correlation_id=properties.correlation_id)

        try:
            req = json.loads(body)
            res = json.dumps({'status': 'empty'})

            if req['method'] == 'transform':
                logging.info('Processing transform RPCTransform for %s, coeff : %s', req['id'], req['coeff'])
                coeff = float(req['coeff'])
                uuid = req['id']
            
                path = os.path.join(self.UPLOAD_FOLDER,f'{uuid}.npy')

                latent = np.load(path)
                logging.info(f"Generating images for {uuid}....")
                original_image = image_to_base64(PIL.Image.open(os.path.join(self.UPLOAD_FOLDER,f'{uuid}.png')))
                transformed_image = move_and_show(latent, self.fatness_direction, coeff, self.generator)
                logging.info(f"Done Generating images for {uuid}....")

                res = json.dumps({
                    'status':'OK',
                    'original_image': original_image,
                    'transformed_image' : image_to_base64(transformed_image)
                })
            else:
                logging.warning("Received unknown method: %s", req['method'])
        except Exception as e :
            logging.warning('RPCTransform Server failed to process request %s', str(e))

            res = json.dumps({
                'status' : 'Problem Occured',
                'error': str(e)
            })

        logging.info('RPCTransform Publishing response: %s', res[:50])

        channel.basic_publish(
            exchange='',
            properties=res_props,
            routing_key=properties.reply_to,
            body=res
        )
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)

        logging.info('RPCTransform Server request finished')

class StatusRpc:
    def __init__(self, config):
        super().__init__()
        self.UPLOAD_FOLDER = config['server'].get('upload_folder')
        
    def on_rx_rpc_request(self, channel, method_frame, properties, body):
        logging.debug('RPCStatus Server processing request: %s', body)

        res_props = pika.BasicProperties(correlation_id=properties.correlation_id)

        try:
            uuid = body.decode()
            path = os.path.join(self.UPLOAD_FOLDER, f"{uuid}.json")
            progress = open(path)
            res = json.dumps(json.load(progress))

        except Exception as e :
            logging.warning('RPCTransform Server failed to process request %s', str(e))

            res =  json.dumps({'status':'Waiting for queue to response','progress':0, 'err': str(e)})


        logging.info('RPCStatus Publishing response: %s', res)

        channel.basic_publish(
            exchange='',
            properties=res_props,
            routing_key=properties.reply_to,
            body=res
        )
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)

        logging.info('RPCStatus Server request finished')


def ProjectionConsumer(args):

    config = ConfigParser()
    config.read(args.config)

    try:
        server = ProjectionRpc(config)
        parameters =  pika.ConnectionParameters(heartbeat=300)
        with pika.BlockingConnection(parameters) as conn:
            channel = conn.channel()

            channel.queue_declare(
                queue=config['server'].get('projection_queue'), 
                exclusive=True, 
                auto_delete=True
            )
            channel.basic_consume(
                config['server'].get('projection_queue'),
                server.on_rx_rpc_request
            )

            logging.info('RPCProjection Server ready.')
            channel.start_consuming()
            
    except Exception as e:
        logging.warning('Caught error:')
        logging.warning(e)

    logging.info('RPCProjection Server shutting down')

def TransformConsumer(args):

    config = ConfigParser()
    config.read(args.config)

    try:
        server = TransformRpc(config)
        parameters =  pika.ConnectionParameters(heartbeat=300)
        with pika.BlockingConnection(parameters) as conn:
            channel = conn.channel()

            channel.queue_declare(
                queue=config['server'].get('transform_queue'), 
                exclusive=True, 
                auto_delete=True
            )
            channel.basic_consume(
                config['server'].get('transform_queue'),
                server.on_rx_rpc_request
            )

            logging.info('RPCTransform Server ready.')
            channel.start_consuming()
            
    except Exception as e:
        logging.warning('Caught error:')
        logging.warning(e)

    logging.info('RPCTransform Server shutting down')

def StatusConsumer(args):

    config = ConfigParser()
    config.read(args.config)

    try:
        server = StatusRpc(config)
        parameters =  pika.ConnectionParameters(heartbeat=300)
        with pika.BlockingConnection(parameters) as conn:
            channel = conn.channel()

            channel.queue_declare(
                queue=config['server'].get('status_queue'), 
                exclusive=True, 
                auto_delete=True
            )
            channel.basic_consume(
                config['server'].get('status_queue'),
                server.on_rx_rpc_request
            )

            logging.info('RPCStatus Server ready.')
            channel.start_consuming()
            
    except Exception as e:
        logging.warning('Caught error:')
        logging.warning(e)

    logging.info('RPCStatus Server shutting down')

class Thread (threading.Thread):
    def __init__(self, args, method):
        threading.Thread.__init__(self)
        self.args = args
        self.method = method

    def run(self):
        if self.method == "Projection":
            ProjectionConsumer(self.args)
        if self.method == "Transform":
            TransformConsumer(self.args)
        if self.method == "Status":
            StatusConsumer(self.args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='settings.conf', type=str)

    logging.getLogger().setLevel('INFO')
    logging.info('RPC Server starting...')

    
    threads = []

    Projection = Thread(parser.parse_args(), "Projection")
    Transform = Thread(parser.parse_args(), "Transform")
    Status = Thread(parser.parse_args(), "Status")


    Projection.start()
    Transform.start()
    Status.start()

    threads.append(Projection)
    threads.append(Transform)
    threads.append(Status)


    for t in threads:
        t.join()
    print ("Exiting Main Thread")

    # main(parser.parse_args())
