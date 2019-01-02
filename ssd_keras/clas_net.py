from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from cls_data_loader import *
from keras import applications
from keras.models import Model

root_data_dir = '/Users/royhirsch/Documents/Study/Current/ComputerVision/project/busesTrain'
label_path = '/Users/royhirsch/Documents/Study/Current/ComputerVision/project/annotationsTrain.txt'
num_classes = 6
dim = 64
vgg_model = applications.VGG16(weights='imagenet',
                               include_top=False,
                               input_shape=(dim, dim, 3))

# Creating dictionary that maps layer names to the layers
layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

# Getting output tensor of the last VGG layer that we want to include
x = layer_dict['block2_pool'].output

# Stacking a new simple convolutional network on top of it
x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(6, activation='softmax')(x)

# Creating new model. Please note that this is NOT a Sequential() model.

custom_model = Model(input=vgg_model.input, output=x)

# Make sure that the pre-trained bottom layers are not trainable
for layer in custom_model.layers[:7]:
    layer.trainable = False

dg = DataGenerator(root_data_dir, label_path,batch_size=8, dim=dim,
	             n_channels=3, n_classes=6, shuffle=True)

model = Sequential()
model.add(Conv2D(32, (5, 5), padding='same', input_shape=(dim, dim, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Conv2D(32, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.adam(lr=1e-3)

# Let's train the model using RMSprop
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])

history = model.fit_generator(generator=dg,
                              steps_per_epoch=100,
                              epochs=10,
                              # callbacks=callbacks,
                              # workers=3,
                              # use_multiprocessing=True,
                              # validation_data=val_generator,
                              # validation_steps=200,  # TODO its' just a number...
                              initial_epoch=0)
