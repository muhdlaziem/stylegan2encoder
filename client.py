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


class FatToThinClient:
    def __init__(self, host, port, queue_name, timeout):
        self.queue_name = queue_name
        self.response = None
        self.corr_id = None
        self.timeout = timeout

        params = pika.ConnectionParameters(host=host, port=port)
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

        
        logging.info('Sending request for file %s', img_path)
        self.corr_id = str(uuid.uuid4())
        with open(img_path, 'rb') as f:
            encoded = json.dumps({
                'method': 'projection',
                'image': base64.b64encode(f.read()).decode('utf-8'),
                'id': self.corr_id
            })
            self.do_rpc(encoded)

        return self.response


def main(args):
    logging.getLogger().setLevel('INFO')
    logging.info('Starting RPC Client...')

    try:
        rpc_client = FatToThinClient(
            args.host,
            args.port,
            args.queue,
            args.timeout
        )
        logging.info('RPC Client connected.')
        results = rpc_client.predict(args.input)

        for result in results:
            logging.info('Received Result: %s', result)

    except Exception as e:
        logging.warning('Caught error:')
        logging.warning(e)

    logging.info('RPC Server shutting down')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost', type=str)
    parser.add_argument('--port', default=5672, type=int)
    parser.add_argument('--queue', default='rpc.ai.FatToThinProjection.queue', type=str)
    parser.add_argument('--timeout', default=350, type=str)
    parser.add_argument('--input', type=str)

    main(parser.parse_args())