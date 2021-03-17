import argparse
from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer
from configparser import ConfigParser
from client import FatToThinClient
from flask_cors import CORS, cross_origin
import os
import json

def main(args):

    config = ConfigParser()
    config.read(args.config)
    app = Flask(__name__)
    app.secret_key = "secret key"
    cors = CORS(app)
    app.config['CORS_HEADERS'] = 'Content-Type'
    UPLOAD_FOLDER = config['server'].get('upload_folder')
    
    @app.route('/projection', methods=['POST'])
    @cross_origin()
    def projection():
        data = request.data
        queue = 'rpc.ai.FatToThinProjection.queue'
        rpc_client = FatToThinClient(
            "localhost",
            5672,
            queue,
            100000,
            'laziem',
            'laziem'
        )
        print('RPC Client connected.')
        results = rpc_client.projection(body=data)

        # print('Received Result: %s', results)
        return results


    @app.route('/transform', methods=['POST'])
    @cross_origin()
    def transform():
        data = request.data
        print(data)
        queue = 'rpc.ai.FatToThinTransform.queue'
        rpc_client = FatToThinClient(
            "localhost",
            5672,
            queue,
            100000,
            'laziem',
            'laziem'
        )
        print('RPC Client connected.')
        results = rpc_client.transform(body=data)

        # print('Received Result: %s', results)
        return results
    
    @app.route('/progress/<uuid>', methods=['POST'])
    @cross_origin()
    def progress(uuid):
        try:
            path = os.path.join(UPLOAD_FOLDER, f"{uuid}.json")
            progress = open(path)
            return jsonify(json.load(progress))
        except Exception as e:
            return jsonify({'status':'Processing projection your request','progress':0, 'err': str(e)})

    host = config['server'].get('host')
    port = int(config['server'].get('port'))

    print('Starting server on host %s port %d' % (host, port))
    app.run(host=host,port=port, threaded=True, debug=True)
    # http_server = WSGIServer((host, port), app)
    # http_server.serve_forever()        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='settings.conf', type=str)

    main(parser.parse_args())
