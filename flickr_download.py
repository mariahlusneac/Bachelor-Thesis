import flickrapi
import urllib.request as req
from PIL import Image
from time import sleep


# Flickr api access key
flickr = flickrapi.FlickrAPI('c6a2c45591d4973ff525042472446ca2', '202ffe6f387ce29b', cache=True)

keyword = 'landscape'

photos = flickr.walk(text=keyword,
                     tag_mode='all',
                     tags=keyword,
                     extras='url_c',
                     per_page=100,  # may be you can try different numbers..
                     sort='relevance')

urls = []
ok = True
for i, photo in enumerate(photos):
    while ok:
        try:
            url = photo.get('url_c')
            if url:
                try:
                    req.urlretrieve(url, f'flickr_photos\\{i}.jpg')
                except flickrapi.exceptions.FlickrError:
                    print('eroare FlickrError ', i)
                    sleep(20)
                except:
                    print('eroare la save ', i)
            urls.append(url)
            print(i, '  ', url)
            ok = False
        except flickrapi.exceptions.FlickrError:
            print('eroare FlickrError ', i)
            sleep(20)
        except:
            print('eroare generala ', i)
    ok = True
    if i > 1000000:
        break


# Download image from the url and save it to '00001.jpg'
# for i in range(len(urls)):
#     if urls[i]:
#         try:
#             req.urlretrieve(urls[i], f'flickr_photos\\{i}.jpg')
#         except:
#             print('exception ', i)

print('done')

# Resize the image and overwrite it
# image = Image.open('00001.jpg')
# image = image.resize((256, 256), Image.ANTIALIAS)
# image.save('00001.jpg')