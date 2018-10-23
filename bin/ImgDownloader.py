import requests

#--Acquire img paths

#--Download img
img_url = 'http://images.fineartamerica.com/images-medium-large-5/abstract-art-original-painting-winter-cold-by-madart-megan-duncanson.jpg'
path = 'data/img/img_url1.jpeg'

r = requests.get(img_url, stream=True)
if r.status_code == 200:
    with open(path, 'wb') as f:
        for chunk in r.iter_content(1024): #'1024 = chunk size
            f.write(chunk)