import requests
import json
import base64
import PIL.Image
from io import BytesIO
url = 'http://3367c75ac0b9.ngrok.io/transform'
myobj = {'id': '5f7082f37f50457f6cd23912bf6fe5a2'}

x = requests.post(url, json = myobj)
data = json.loads(x.text)
decoded = base64.b64decode(data['transformed_image'])



img = PIL.Image.open(BytesIO(decoded))
img.save('result.png')