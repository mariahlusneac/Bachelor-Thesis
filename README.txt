Seturile de date au fost preluate de pe:
https://www.kaggle.com/jessicali9530/celeba-dataset
http://places.csail.mit.edu/
Acestea nu au fost utilizate in intregime in lucrarea de licenta si, de asemenea, imaginile din primul set de date au fost usor modificate. Seturile de date au fost utilizate sub forma de array-uri Numpy, care ocupa mult spatiu si, din acest motiv, nu au fost incarcate in prezenta arhiva si nici pe github. Modelele antrenate nu au fost incarcate din acelasi motiv.




Fisiere sursa

flickr_download.py - descarcarea unor peisaje de pe Flickr
flickr_resize.py - redimensionarea peisajelor descarcate de pe Flickr
take_ss_for_youtube_videos.py - realizarea de snapshot-uri ale ecranului in vederea obtinerii de peisaje din videoclipuri de pe Youtube
create_portraits_dataset.ipynb - crearea setului de date cu portrete
create_landscapes_dataset.py - crearea setului de date cu peisaje

classifier.ipynb - retea de clasificare a portretelor si peisajelor

training_portraits_autoenc_gan.py - retelele autoencoder si GAN pentru portrete
training_autoenc_small_landscapes.py - autoencoder pentru setul restrans de peisaje
training_autoenc_big_landscapes.py - autoencoder pentru setul extins de peisaje
training_gan_small_landscapes.py - gan pentru setul restrans de peisaje
training_gan_big_landscapes.py - gan pentru setul extins de peisaje

predict_postprocess_plot.ipynb - predictie (dupa antrenare), postprocesarea imaginilor si realizarea de grafice

portraits_compute_metric.ipynb - calcularea valorii metricii pe rezultatele de la portrete
landscapes_compute_metric.ipynb - calcularea valorii metricii pe rezultatele de la peisaje

LICENSE.txt - licenta GPLv3
