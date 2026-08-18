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



def downloadDataSet(searches, path):
    for o in searches:
            dest = (path/o)
            dest.mkdir(exist_ok=True, parents=True)
            download_images(dest, urls=search_img(f'{o} photo'))
            time.sleep(2)
            download_images(dest, urls=search_img(f'{o} sun photo'))
            time.sleep(2)
            download_images(dest, urls=search_img(f'{o} shade photo'))
            time.sleep(2)
            #resize images
            resize_images(path/o, max_size=400, dest=path/o, max_workers=0, recurse=True)


def cleanDataSet(path):
    fns = get_image_files(path)
    #print(fns)
    
    #clean file with verify_image and not verify_images as 64 cores break code
    for fn in fns:
        passed = verify_image(fn)
        if not passed:
            os.unlink(fn)


def trainOnDataSet(dls, epoch=3):
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(epoch)
    return learn


def main():
    #create data set
    searches = 'bird','forest'
    path = Path('bird_or_not')


    downloadDataSet(searches, path)
    print("data set downloaded")

    cleanDataSet(path)
    print("data set cleaned")


    #create DataBlock, need num_workers fixed to avoid core > 63 causing crash
    dls = DataBlock(
        #input are images block, outputs are categories
        blocks=(ImageBlock, CategoryBlock),
        #get all images from path
        get_items=get_image_files,
        #keep 20% for cross validation
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        #set y value as the name of our folder
        get_y=parent_label,
        #rseize image to 192*192 and squish them not crop to keep all informations
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(path, bs=32, num_workers=0)

    #dls.show_batch(max_n=6)

    #train the model on resnet18 with 3 epoch
    learn = trainOnDataSet(dls, 3)

    #try model on local bird.jpg
    is_bird,_,probs = learn.predict(PILImage.create('bird.jpg'))
    print(f"This is a: {is_bird}.")
    print(f"Probability it's a bird: {probs[0]:.4f}")


if __name__ == '__main__':
    main()