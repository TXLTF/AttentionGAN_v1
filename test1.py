import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

from PIL import Image
import numpy as np


def split_image(image, patch_size):
    """
    Split an image into smaller patches of a given size.
    """
    image = np.array(image)
    patches = []
    for i in range(0, image.shape[0], patch_size):
        for j in range(0, image.shape[1], patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append((patch, (i, j)))
    return patches


def stitch_patches(patches, image_shape, patch_size):
    """
    Stitch the patches back into the original image shape.
    """
    stitched_image = np.zeros(image_shape, dtype=np.uint8)
    for patch, (i, j) in patches:
        stitched_image[i:i + patch_size, j:j + patch_size] = patch
    return Image.fromarray(stitched_image)


def process_image_with_patches(model, image, patch_size):
    """
    Process the image by splitting it into patches, running inference, and stitching it back together.
    """
    patches = split_image(image, patch_size)
    processed_patches = []

    for patch, (i, j) in patches:
        patch_image = Image.fromarray(patch)
        model.set_input({'A': patch_image})  # Adjust input data structure as needed
        model.test()
        patch_result = model.get_current_visuals()['fake_B']  # Adjust output key as needed
        processed_patches.append((patch_result, (i, j)))

    return stitch_patches(processed_patches, image.size, patch_size)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

    # patch_size = 256  # Define the size of the patches here

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    if opt.eval:
        model.eval()

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break

        print("data:",data)
        model.set_input(data)  # unpack data from data loader

        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths


        # # Process image with patches
        # input_image = data['A']  # Adjust this as needed
        # processed_image = process_image_with_patches(model, input_image, patch_size)
        #
        # # Save the results
        # visuals = {'fake_B': processed_image}  # Adjust this as needed
        # img_path = model.get_image_paths()  # Adjust this as needed

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()  # save the HTML
