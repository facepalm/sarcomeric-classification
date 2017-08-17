import skimage.measure
import numpy as np
import random

window = [150,150]
GAP = 10
BATCH = 16
EPOCHS = 10

files = ['Example1.col.jpg',
         'Example2.col.jpg',
         'Example3.col.jpg',
         'Example4.col.jpg',
         'Example5.col.jpg',
         'Example6.col.jpg',
         'Example7.col.jpg']

def init_ML_model():
    from keras.applications.vgg16 import VGG16
    import keras.applications
    from keras.preprocessing import image
    from keras.models import Model
    from keras.applications.vgg16 import preprocess_input
    from keras.layers import Dense, GlobalAveragePooling2D, Input, RepeatVector

    input_tensor = Input(shape=(window[0], window[1],3))  # this assumes K.image_data_format() == 'channels_last'

    base_model = VGG16(input_tensor = input_tensor, weights='imagenet', include_top=False)
    #base_model = keras.applications.resnet50.ResNet50(input_tensor = input_tensor, weights='imagenet', include_top=False)
    #base_model = keras.applications.inception_v3.InceptionV3(input_tensor = input_tensor, weights='imagenet', include_top=False)

    x = base_model.output
    # add a global spatial average pooling layer NOT RIGHT NOW TEST LATER
    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = Dense(100, activation='relu')(x)
    # and a logistic layer -- with 2 classes (discontinuous/not disc)
    predictions = Dense(2, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    return model

def make_ML_training_data(image, mask, balance=False):
    import keras.utils
    import random

    smallshape = (np.array(image.shape) - np.array(window)+[GAP,GAP])/GAP
    data = np.zeros((smallshape[0]*smallshape[1],window[0],window[1],3),dtype = np.uint16)
    labels = []

    print smallshape[0]*smallshape[1], image.shape[0]-window[0]/2, image.shape[1]-window[1]/2

    curr = 0
    for y in np.arange(window[0]/2,image.shape[0]-window[0]/2,GAP):
        for x in np.arange(window[1]/2,image.shape[1]-window[1]/2,GAP):
            if balance and not mask[y,x] and random.random() < 0.95: continue
            patch = image[y-window[0]/2:y+window[0]/2,x-window[1]/2:x+window[1]/2]
            patch = np.expand_dims(patch, axis=2)
            data[curr,:,:,:] = patch.repeat(3,2)
            labels.append(mask[y,x])
            curr += 1

    print curr
    data = data[0:curr,:,:,:]

    # Convert labels to categorical one-hot encoding
    one_hot_labels = keras.utils.to_categorical(labels, num_classes=2)

    print one_hot_labels.sum(axis=0)

    return data, one_hot_labels


def train_ML(model, data, labels,name=None):

    # Train the model, iterating on the data in batches of 32 samples
    model.fit(data, labels, epochs = EPOCHS, batch_size = BATCH)

    if not name: name = 'my_model.h5'

    model.save(name)  # creates a HDF5 file 'my_model.h5'

    return model


def load_ML(name='my_model.h5'):
    from keras.models import load_model

    # returns a compiled model
    # identical to the previous one
    model = load_model( name )
    return model

def do_ML_processing():
    import skimage.io
    import skimage.transform
    import skimage.util

    use_network = True
    load_network = True
    train_network = False

    if train_network:
        data, labels = None, None
        for f in files:
            print 'Processing ',f
            img = skimage.io.imread(f)
            img_data = skimage.measure.block_reduce( img[:,:,1].squeeze() , block_size=(2, 2), func=np.mean)
            img_mask = skimage.measure.block_reduce( img[:,:,0] > img[:,:,1] , block_size=(2, 2), func=np.mean)

            tdata,tlabels = make_ML_training_data(img_data, img_mask, balance=True)

            data = tdata if data is None else np.concatenate((data, tdata), axis = 0)
            labels = tlabels if labels is None else np.concatenate((labels, tlabels), axis = 0)

        print data.shape, labels.shape
        print 'Data prepped, training model'

    if use_network:
        if load_network:
            model_file = 'my_model.h5'
            model = load_ML(model_file)
        else:
            model = init_ML_model()
        if train_network:
            model = train_ML( model, data, labels )

    data, labels = None, None
    for f in [files[1]]:
        print 'Processing ',f

        img = skimage.io.imread(f)

        img_data = skimage.measure.block_reduce( img[:,:,1].squeeze() , block_size=(2, 2), func=np.mean)
        img_mask = skimage.measure.block_reduce( img[:,:,0] > img[:,:,1] , block_size=(2, 2), func=np.mean)
        labelshape = (np.array(img_data.shape) - np.array(window)+[GAP,GAP])/GAP

        tdata,tlabels = make_ML_training_data(img_data, img_mask)

        data = tdata if data is None else np.concatenate((data, tdata), axis = 0)
        labels = tlabels if labels is None else np.concatenate((labels, tlabels), axis = 0)

    print labels.shape
    if use_network:
        labels = model.predict(data, batch_size=BATCH, verbose = 1)
    else:
        labels = np.random.random((6966,2))

    yes_label = np.reshape(labels[:,1],labelshape)

    offset = [window[0]/2 - GAP/2, window[1]/2 - GAP/2]

    print yes_label.shape, np.multiply(yes_label.shape,GAP)
    yes_label = skimage.transform.resize(yes_label, np.multiply(yes_label.shape,GAP), order=0 )

    yes_label = skimage.util.pad(yes_label,((offset[0],img_data.shape[0]-(yes_label.shape[0]+offset[0])),(offset[1],img_data.shape[1]-(yes_label.shape[1]+offset[1]))), 'constant', constant_values=(0))

    print yes_label.shape, img_data.shape

    skimage.io.imsave('labels.tif',yes_label.astype(np.float32))


def spawn_templates():
    import template_functions
    templates = []

    for period in [120, 100, 80]:
        for stagger in [-0.5,0.5,1]:
            for tiltu in [-0.5,0,0.5]:
                for tiltd in [-0.5,0,0.5]:
                    for cont in [0.4, 0.7, 1.0]:
                        templates.append( spawn_a_template(period/GAP,stagger,tiltu,tiltd,cont) )

    return templates


def spawn_a_template(period=120,stagger=1,tiltu=0,tiltd=0, contrast=1.):
    import template_functions
    templ = template_functions.createTemplate(period,stagger,[tiltu,tiltd], contrast)
    return templ


def apply_templates(img, mask,templates, balance = True):
    import skimage.feature

    tlabels = mask.flatten().astype(np.uint8)
    outscores = None
    for t in templates:
        tscore = np.expand_dims( np.abs( skimage.feature.match_template(img,t,pad_input=True) ).flatten() , 1)
        outscores = tscore if outscores is None else np.concatenate((outscores, tscore), axis = 1)

    keep = np.zeros(tlabels.shape)
    keep[ tlabels == 1 ] = 1
    for i in range(keep.shape[0]):
        if not balance or tlabels[i] == 0 and random.random() > 0.95:
            keep[i] = 1

    outscores = outscores[keep == 1 , :]
    tlabels = tlabels[keep == 1]

    return outscores, tlabels


def do_template_processing():
    import skimage.io
    import skimage.transform
    import skimage.feature
    import skimage.util
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.metrics import zero_one_loss
    from sklearn.naive_bayes import GaussianNB

    templates = spawn_templates()

    n_estimators = 100

    data, labels = None, None

    img = skimage.io.imread(files[5])
    img_data = skimage.measure.block_reduce( img[:,:,1].squeeze() , block_size=(GAP, GAP), func=np.mean)
    vern_templ = spawn_a_template(period=120/GAP, contrast = 0.5)
    match = skimage.feature.match_template(img_data,vern_templ,pad_input=True)
    skimage.io.imsave('test.tif',np.abs(match).astype(np.float32))
    skimage.io.imsave('test_template.tif',vern_templ.astype(np.float32))

    for f in files:
        print 'Processing ',f
        img = skimage.io.imread(f)
        img_data = skimage.measure.block_reduce( img[:,:,1].squeeze() , block_size=(GAP, GAP), func=np.mean)
        img_mask = skimage.measure.block_reduce( img[:,:,0] > img[:,:,1] , block_size=(GAP, GAP), func=np.mean)

        tdata, tlabels = apply_templates(img_data, img_mask, templates)
        print tdata.shape, tlabels.shape, tlabels.mean()

        data = tdata if data is None else np.concatenate((data, tdata), axis = 0)
        labels = tlabels if labels is None else np.concatenate((labels, tlabels), axis = 0)

    print data.shape, labels.shape

    indices = np.arange(labels.shape[0])
    np.random.shuffle(indices)

    decimate = labels.shape[0]/10
    testx, testy = data[indices[:decimate],:], labels[indices[:decimate]]
    trainx, trainy = data[indices[decimate:],:], labels[indices[decimate:]]

    #np.random.shuffle( labeled_data )

    print trainx.shape, trainy.shape, testx.shape, testy.shape

    #Train ml
    gauss = GaussianNB()
    ada = AdaBoostClassifier( n_estimators = n_estimators )

    classifier = ada

    #ada.fit( trainx, trainy )
    classifier.fit( trainx, trainy )
    score = classifier.score( testx, testy )
    print score

    img = skimage.io.imread(files[1])
    img_data = skimage.measure.block_reduce( img[:,:,1].squeeze() , block_size=(GAP, GAP), func=np.mean)
    img_mask = skimage.measure.block_reduce( img[:,:,0] > img[:,:,1] , block_size=(GAP, GAP), func=np.mean)

    tdata, tlabels = apply_templates(img_data, img_mask, templates, balance = False)

    predict = classifier.predict_proba( tdata )
    print predict
    predict = np.reshape(predict[:,1],img_mask.shape)

    skimage.io.imsave('test_prediction_2.tif',predict.astype(np.float32))

    ada_err = np.zeros((n_estimators,))
    for i, y_pred in enumerate(classifier.staged_predict(testx)):
        ada_err[i] = zero_one_loss(y_pred, testy)
    print ada_err


if __name__ == '__main__':
    do_ML_processing()
    #do_template_processing()
