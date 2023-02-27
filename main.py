from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

print("-----------")

data_gen = ImageDataGenerator(rescale=1.0/255)

imgdir = 'a5_images' # or wherever you put them...
img_size = 64
batch_size = 32

train_generator = data_gen.flow_from_directory(
        imgdir + '/train',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        classes=['other', 'car'],
        seed=12345,
        shuffle=True)



Xbatch, Ybatch = train_generator.next()

print(Xbatch.shape)


plt.imshow(Xbatch[4])
plt.show()

