# -*- coding: utf-8 -*-

# Sample code to use string producer.

import tensorflow as tf
import numpy as np


def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    """if type(x) == list:
        x = np.array(x)
    x = x.flatten()"""
    o_h = np.zeros(n)
    """quito el len(x) porque no es compatible con float"""
    o_h[x] = 1
    return o_h


num_classes = 3 #cambiar numero de clases de 2 a 3
batch_size = 10

# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot(int(i), num_classes)  # [one_hot(float(i), num_classes)] , cambio a int float da error
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 3, 18 * 33 * 64]), units=35, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax) #cambiar sigmoide a softmax y el numero de unidades a 3
    return y


example_batch_train, label_batch_train = dataSource(["data3/0_train/*.jpg", "data3/1_train/*.jpg", "data3/2_train/*.jpg"], batch_size=batch_size)
example_batch_valid, label_batch_valid = \
    dataSource(["data3/0_valid/*.jpg", "data3/1_valid/*.jpg", "data3/2_valid/*.jpg"], batch_size=batch_size)
example_batch_test, label_batch_test = \
    dataSource(["data3/0_test/*.jpg", "data3/1_test/*.jpg", "data3/2_test/*.jpg"], batch_size=batch_size)


example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)


cost = tf.reduce_sum(tf.square(example_batch_train_predicted - tf.cast(label_batch_train, dtype=tf.float32)))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - tf.cast(label_batch_valid, dtype=tf.float32)))
cost_test = tf.reduce_sum(tf.square(example_batch_test_predicted - tf.cast(label_batch_test, dtype=tf.float32)))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    """while como practica 2"""
    i = 0
    while i <= 100:
        sess.run(optimizer)
        i = i+1
    errorActual = 100
    errorAnterior = 0
    comp = 10
    epoch = 0
    error = []

    while comp > 0.05:  # si la diferencia entre los errores es menor a un 2% para
        sess.run(optimizer)
        errorAnterior = errorActual
        errorActual = sess.run(cost_valid)
        error.append(errorActual)
        print "El error actual es", errorActual
        comp = abs(errorActual - errorAnterior) / errorAnterior
        epoch = epoch + 1
        print "Epoch #:", epoch, "Error: ", errorActual
        print "----------------------------------------------------------------------------------"
    print "///////////////////////////////////---TEST---///////////////////////////////////////////////"
    obtenido = sess.run(example_batch_test_predicted)
    perfecto = sess.run(label_batch_test)
    bien = 0
    mal = 0
    bienAnterior = 0

    for b, r in zip(perfecto, obtenido):
        if (np.argmax(b) == np.argmax(r)):
            bien = bien + 1
        else:
            mal = mal + 1

        print (b, "-->", r)

        if (bienAnterior < bien):
            print "ACIERTA"
        else:
            print "FALLA"
        bienAnterior = bien
        malAnterior = mal

    print "Aciertos totales ", bien
    print "Fallos totales: ", mal
    todos = bien + mal

    print "Se ha acertado en el ", (float(bien) / float(todos)) * 100, "% de los casos"

    import matplotlib.pyplot as plt

    plt.plot(error)
    plt.legend(['Evolucion del error'])
    plt.show()  # Let's see a sample

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)
    coord.request_stop()
    coord.join(threads)