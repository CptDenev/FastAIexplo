from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import *

from itertools import islice
from ddgs import DDGS

import time

#search a fixed image number for a given term
def search_img(term, max_images=10):
    print(f"Searching for '{term}'")
    # DDGS().images return an interator containing all images found
    # we use islice to limit the number of results returned
    return L(islice(DDGS().images(term), max_images)).itemgot('image')



#create data set
searches = 'bird','forest'
path = Path('bird_or_not')

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_img(f'{o} photo'))
    time.sleep(2)
    '''
    download_images(dest, urls=search_images(f'{o} sun photo'))
    time.sleep(2)
    download_images(dest, urls=search_images(f'{o} shade photo'))
    time.sleep(2)
    '''
    #resize_images(path/o, max_size=400, dest=path/o, max_workers=0, recurse=True)




print("set downloaded")
fns = get_image_files(path)
print(fns)

#clean file with verify_image and not verify_images as 64 cores break code
for fn in fns:
    passed = verify_image(fn)
    if not passed:
        os.unlink(fn)



#create DataBlock
dls = DataBlock(
    #input are images block, outputs are categories
    blocks=(ImageBlock, CategoryBlock),
    #get all images from path
    get_items=get_image_files,
    #keep 20% for cross validation
    splitter=RandomSplitter(valid_pct=0.2, seed=42)
    #set y value as the name of our folder
    get_y=parent_label,
    #rseize image to 192*192 and squish them not crop to keep all informations
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)



#train the model on resnet18
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)