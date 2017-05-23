import os
from PIL import Image
import tensorflow as tf

def create_tfrecords(file_name, records_name):

    f = open(file_name)
    writer = tf.python_io.TFRecordWriter(records_name)
    step = 0
    for line in f.readlines():
        items = line.split()
        img_path = items[1]
        label = int(items[0])

        img = Image.open(img_path)
        img_width, img_height = img.size[:2]
        img_raw = img.tobytes()

        example = tf.train.Example(features = tf.train.Features(feature = {
            'label' : tf.train.Feature(int64_list = tf.train.Int64List(value = [label])),
            'height' : tf.train.Feature(int64_list = tf.train.Int64List(value = [img_height])),
            'width' : tf.train.Feature(int64_list = tf.train.Int64List(value = [img_width])),
            'img_raw' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [img.tobytes()]))
        }))

        writer.write(example.SerializeToString())
        step += 1
        print(step)


# prepare tfrecord
#list_file = "digit_data/testImages/labels.txt"
#create_tfrecords(list_file, "test.tfrecords")
train_file = "digit_data/trainImages/labels.txt"
create_tfrecords(train_file, 'train.tfrecords')

for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    image = example.features.feature['image'].bytes_list.value
    label = example.features.feature['label'].int64_list.value
    w = example.features.feature['width'].int64_list.value
    h = example.features.feature['height'].int64_list.value
    print(image, label, w, h)
