import requests
import json
import base64
import PIL.Image
from io import BytesIO

def image_to_base64(img):
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str.decode()

image = PIL.Image.open('/home/muhdlaziem/Downloads/TQ7A9056.jpg')
b64 = image_to_base64(image)
url = 'http://8b4273cd78d6.ngrok.io/projection'
# transform = {'id': '6ab6343e6316074324f57003196aad8f', 'coeff' : '-0.8'}
encode = {'image' : b64}
x = requests.post(url, json = encode)
print(x.text)
# data = json.loads(x.text)
# decoded = base64.b64decode(data['transformed_image'])



# img = PIL.Image.open(BytesIO(decoded))
# img.show()