import argparse
import base64
from configparser import ConfigParser
import io
import time
import json
import logging
import os
import pika
import uuid
from PIL import Image
from io import BytesIO

class FatToThinClient:
    def __init__(self, host, port, queue_name, timeout, username, password):
        self.queue_name = queue_name
        self.response = None
        self.corr_id = None
        self.timeout = timeout

        credentials = pika.PlainCredentials(username, password)
        params = pika.ConnectionParameters(credentials=credentials)
        # parameters = pika.URLParameters('amqp://laziem:laziem@localhost:5672/%2F')
        self.connection = pika.BlockingConnection(params)
        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True
        )

    def on_response(self, _ch, _method, props, body):
        self.response = body

    def do_rpc(self, body):
        deadline = time.time() + self.timeout
        self.channel.basic_publish(
            exchange='',
            routing_key=self.queue_name,
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=body)
        while self.response is None and time.time() < deadline:
            self.connection.process_data_events()

    def projection(self, img_path):
        self.response = None

        self.corr_id = str(uuid.uuid4())
        
        logging.info('Sending request for file %s : %s ', img_path, self.corr_id)
        with open(img_path, 'rb') as f:
            encoded = json.dumps({
                'method': 'projection',
                'image': base64.b64encode(f.read()).decode('utf-8'),
                'id': self.corr_id
            })
            self.do_rpc(encoded)

        return self.response
    
    def transform(self, uuid, coeff):
        self.response = None        
        logging.info('Sending request for file %s : %s ', uuid, coeff)
        encoded = json.dumps({
            'method': 'transform',
            'id': uuid,
            'coeff': coeff
        })
        self.do_rpc(encoded)
        data = json.loads(self.response)
        decoded = base64.b64decode(data['transformed_image'])
        img = Image.open(BytesIO(decoded))
        img.show()
        return data['status']


def main(args):
    logging.getLogger().setLevel('INFO')
    logging.info('Starting RPC Client...')
    queue = None

    try:
        if args.tf:
            queue = 'rpc.ai.FatToThinTransform.queue'
            rpc_client = FatToThinClient(
                args.host,
                args.port,
                queue,
                args.timeout,
                args.username,
                args.password
            )
            logging.info('RPC Client connected.')
            results = rpc_client.transform(args.id, args.coeff)

            logging.info('Received Result: %s', results)

        if args.en:
            queue = 'rpc.ai.FatToThinProjection.queue'
            rpc_client = FatToThinClient(
                args.host,
                args.port,
                queue,
                args.timeout,
                args.username,
                args.password
            )
            logging.info('RPC Client connected.')
            results = rpc_client.projection(args.input)

            logging.info('Received Result: %s', results)

       

    except Exception as e:
        logging.warning('Caught error:')
        logging.warning(e)

    logging.info('RPC Server shutting down')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost', type=str)
    parser.add_argument('--port', default=5672, type=int)
    parser.add_argument('--timeout', default=1000000, type=str)
    parser.add_argument('--input', type=str)
    parser.add_argument('--username', default='laziem', type=str)
    parser.add_argument('--password', default='laziem', type=str)
    parser.add_argument('--tf', default=False, action='store_true')
    parser.add_argument('--en', default=False, action='store_true')
    parser.add_argument('--id', default='ee1d97fc-66ab-42e9-ba03-d77172564cda', type=str)
    parser.add_argument('--coeff', default='0.7', type=str)

    main(parser.parse_args())