import tensorflow as tf

def cutmix(image, label, batch_size, image_size, num_classes, PROBABILITY=1.0):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with cutmix applied
    DIM = image_size

    imgs = []
    labs = []
    for j in range(batch_size):
        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)
        # CHOOSE RANDOM IMAGE TO CUTMIX WITH
        k = tf.cast(tf.random.uniform([], 0, batch_size), tf.int32)
        # CHOOSE RANDOM LOCATION
        x = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        b = tf.random.uniform([], 0, 1)  # this is beta dist with alpha=1.0
        WIDTH = tf.cast(DIM * tf.math.sqrt(1 - b), tf.int32) * P
        ya = tf.math.maximum(0, y - WIDTH // 2)
        yb = tf.math.minimum(DIM, y + WIDTH // 2)
        xa = tf.math.maximum(0, x - WIDTH // 2)
        xb = tf.math.minimum(DIM, x + WIDTH // 2)
        # MAKE CUTMIX IMAGE
        one = image[j, ya:yb, 0:xa, :]
        two = image[k, ya:yb, xa:xb, :]
        three = image[j, ya:yb, xb:DIM, :]
        middle = tf.concat([one, two, three], axis=1)
        img = tf.concat([image[j, 0:ya, :, :], middle, image[j, yb:DIM, :, :]], axis=0)
        imgs.append(img)
        # MAKE CUTMIX LABEL
        a = tf.cast(WIDTH * WIDTH / DIM / DIM, tf.float32)
        if len(label.shape) == 1:
            lab1 = tf.one_hot(label[j], num_classes)
            lab2 = tf.one_hot(label[k], num_classes)
        else:
            lab1 = label[j,]
            lab2 = label[k,]
        labs.append((1 - a) * lab1 + a * lab2)

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs), (batch_size, DIM, DIM, 3))
    label2 = tf.reshape(tf.stack(labs), (batch_size, num_classes))
    return image2, label2


def mixup(image, label, batch_size, image_size, num_classes, PROBABILITY=1.0):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with mixup applied
    DIM = image_size

    imgs = []
    labs = []
    for j in range(batch_size):
        # DO MIXUP WITH PROBABILITY DEFINED ABOVE
        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.float32)
        # CHOOSE RANDOM
        k = tf.cast(tf.random.uniform([], 0, batch_size), tf.int32)
        a = tf.random.uniform([], 0, 1) * P  # this is beta dist with alpha=1.0
        # MAKE MIXUP IMAGE
        img1 = image[j,]
        img2 = image[k,]
        imgs.append((1 - a) * img1 + a * img2)
        # MAKE CUTMIX LABEL

        lab1 = label[j,]
        lab2 = label[k,]
        labs.append((1 - a) * lab1 + a * lab2)

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)

    image2 = tf.reshape(tf.stack(imgs), (batch_size, DIM, DIM, 3))
    label2 = tf.reshape(tf.stack(labs), (batch_size, num_classes))
    return image2, label2


def cut_mix_and_mix_up(image_batch, label_batch, batch_size, image_size, num_classes):
    # THIS FUNCTION APPLIES BOTH CUTMIX AND MIXUP
    DIM = image_size

    SWITCH = 0.5
    CUTMIX_PROB = 0.666
    MIXUP_PROB = 0.666
    # FOR SWITCH PERCENT OF TIME WE DO CUTMIX AND (1-SWITCH) WE DO MIXUP
    image2, label2 = cutmix(image_batch, label_batch, batch_size, image_size, num_classes, CUTMIX_PROB)
    image3, label3 = mixup(image_batch, label_batch, batch_size, image_size, num_classes, MIXUP_PROB)
    imgs = []
    labs = []
    for j in range(batch_size):
        P = tf.cast(tf.random.uniform([], 0, 1) <= SWITCH, tf.float32)
        imgs.append(P * image2[j,] + (1 - P) * image3[j,])
        labs.append(P * label2[j,] + (1 - P) * label3[j,])
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image4 = tf.reshape(tf.stack(imgs), (batch_size, DIM, DIM, 3))
    label4 = tf.reshape(tf.stack(labs), (batch_size, num_classes))
    return image4, label4