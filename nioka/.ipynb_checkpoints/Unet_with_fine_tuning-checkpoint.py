# from https://github.com/killthekitten/kaggle-carvana-2017/blob/master/models.py

from keras.applications.vgg16 import VGG16
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D

from inception_resnet_v2 import InceptionResNetV2
from mobile_net_fixed import MobileNet
from resnet50_fixed import ResNet50
# from param import args

import Unet_with_fine_tuning_models
import losses
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
from skimage.transform import rescale
from scipy.misc import imresize
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
# from skimage.util.montage import montage2d as montage
from skimage.morphology import binary_opening, disk
from sklearn.model_selection import train_test_split
from skimage.morphology import label
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm
from glob import glob
from PIL import ImageFile,Image
import losses
import gc; gc.enable()

image_rgb_dir = "./Original_image/nerve_split/"
image_mask_dir = "./Binary_image/nerve_split/"

input_shape = (256,256,1)

train_list = glob("./Original_image/nerve_split/*.bmp")
tmp=[]
for id in train_list:
    tmp.append(id.split('/')[-1])
    # print(id)

train_list=tmp
train_list, valid_list = train_test_split(train_list,test_size=0.1)



"""         Decode RLEs into Images         """


def make_image_gen(in_list, batch_size):
    all_batches = in_list
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id in all_batches:
            rgb_path = image_rgb_dir+c_img_id
            c_img = imread(rgb_path)
            c_img = np.reshape(c_img,(c_img.shape[0],c_img.shape[1],1))
            rgb_path=rgb_path.split('/')[-1]
            name, ext = os.path.splitext(rgb_path)
            mask_path = image_mask_dir+name+'_mask'+ext
            # print(mask_path)
            c_mask = imread(mask_path)
            c_mask = np.reshape(c_mask,(c_mask.shape[0],c_mask.shape[1],1))
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)/255.0
                out_rgb, out_mask=[], []

"""         Augmentation            """


dg_args = dict(featurewise_center = False,
               samplewise_center = False,
               rotation_range = 45,
               width_shift_range = 0.1,
               height_shift_range = 0.1,
               shear_range = 0.01,
               zoom_range = [0.9, 1.1],
               horizontal_flip = True,
               vertical_flip = True,
               fill_mode = 'reflect',
               data_format = 'channels_last')

image_gen = ImageDataGenerator(**dg_args)
label_gen = ImageDataGenerator(**dg_args)


def create_aug_gen(in_gen, seed = None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(255*in_x,
                             batch_size = in_x.shape[0],
                             seed = seed,
                             shuffle=True)
        g_y = label_gen.flow(in_y,
                             batch_size = in_x.shape[0],
                             seed = seed,
                             shuffle=True)

        yield next(g_x)/255.0, next(g_y)


# t_x, t_y = next(create_aug_gen(train_gen))
gc.collect()

"""         Build a Model           """


make_model = Unet_with_fine_tuning_models
model_name = 'simple_unet'     # resnet50, inception_resnet_v2, mobilenet, vgg, simple_unet
model = make_model.chose_model(input_shape,model_name)

make_loss = losses
model.compile(optimizer=Adam(1e-3, decay=1e-6), loss=make_loss.dice_coef_loss, metrics=['accuracy', make_loss.dice_coef])

weight_path="{}_weights.best.hdf5".format('seg_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min',
                             save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                   patience=1, verbose=1, mode='min',
                                   epsilon=0.0001, cooldown=2, min_lr=1e-7)

early = EarlyStopping(monitor="val_loss", mode="min", verbose=2,
                      patience=20) # probably needs to be more patient, but kaggle time is limited

callbacks_list = [checkpoint, early, reduceLROnPlat]

# callbacks_list = [checkpoint, reduceLROnPlat]



valid_x, valid_y = next(make_image_gen(valid_list,batch_size=len(valid_list)))

BATCH_SIZE = 16

# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 7
MAX_TRAIN_EPOCHS = 99

epoch = min(MAX_TRAIN_STEPS, len(train_list)//BATCH_SIZE)
aug_gen = create_aug_gen(make_image_gen(train_list,BATCH_SIZE))
loss_history = [model.fit_generator(aug_gen,
                                    steps_per_epoch=epoch,
                                    epochs=MAX_TRAIN_EPOCHS,
                                    validation_data=(valid_x, valid_y),
                                    callbacks=callbacks_list,
                                    # workers=1 # the generator is not very thread safe
                                    verbose=1
                                   )]


def save_loss(loss_history):
    epich = np.cumsum(np.concatenate(
        [np.linspace(0.5, 1, len(mh.epoch)) for mh in loss_history]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    _ = ax1.plot(epich,
                 np.concatenate([mh.history['loss'] for mh in loss_history]),
                 'b-',
                 epich, np.concatenate(
            [mh.history['val_loss'] for mh in loss_history]), 'r-')
    ax1.legend(['Training', 'Validation'])
    ax1.set_title('Loss')

    fig.savefig('result.png')


save_loss(loss_history)

model.load_weights(weight_path)
model.save('model_unet_with_'+model_name+'.h5')



##############################################
#               predict
##############################################

img_list = glob("./Original_image/nerve_split/*.bmp")
for img_id in img_list:
    img = imread(img_id)
    img=np.reshape(img,(input_shape[0],input_shape[1],1)).astype(np.float)
    img/=255.
    img=np.expand_dims(img,axis=0)
    img_mask=model.predict(img)
    # print(img_mask.shape)
    img_mask*=255.0
    img_mask=np.reshape(img_mask,(input_shape[0],input_shape[1])).astype(np.uint8)
    # print(img_mask)
    img_mask[img_mask >= 127.5]=255
    img_mask[img_mask <127.5]=0
    result_img = Image.fromarray(img_mask)
    c_img_id = img_id.split('/')[-1]
    name, ext = os.path.splitext(c_img_id)
    result_img.save('./result/' + name + '_mask_unet'+ext)
    # print('./result/' + name + '_mask_'+ext)


