import requests
import json
import base64
import PIL.Image
from io import BytesIO

import argparse

def image_to_base64(img):
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str.decode()

def test_transform(host, id, coeff):
    url = host + "transform"
    transform = {'id': id, 'coeff' : coeff}
    x = requests.post(url, json = transform)
    data = json.loads(x.text)
    decoded = base64.b64decode(data['transformed_image'])
    img = PIL.Image.open(BytesIO(decoded))
    img.show()


def test_encode(host, b64image):
    url = host + "projection"
    encode = {'image' : b64image}
    x = requests.post(url, json = encode)
    print(x.text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--tf', default=False, action='store_true')
    parser.add_argument('--en', default=False, action='store_true')
    parser.add_argument('--host', default='', type=str)
    parser.add_argument('--id', default='5f7082f37f50457f6cd23912bf6fe5a2', type=str)
    parser.add_argument('--coeff', default='0.7', type=str)
    parser.add_argument('--img', default='', type=str)

    args = parser.parse_args()

    if(args.tf):
        test_transform(args.host, args.id, args.coeff)
    elif(args.en):
        img = PIL.Image.open(args.img)
        b64image = image_to_base64(img)
        test_encode(args.host, b64image)
    else:
        print('OPTION NOT FOUND')

