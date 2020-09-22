import scipy.misc
import random

# Inputs
xs1 = []
xs2 = []

# Outputs
ys1 = []
ys2 = []
ys3 = []


#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

#read data.txt
with open("/mrtstorage/users/rehman/datasets/driving_dataset/data.txt") as f:
    for line in f:
        xs1.append("/mrtstorage/users/rehman/datasets/driving_dataset/" + line.split()[0])
        xs2.append("/mrtstorage/users/rehman/datasets/driving_dataset/" + line.split()[0])

        ys1.append(float(line.split()[1]))
        ys2.append(float(line.split()[1]))
        ys3.append(float(line.split()[1]))
        


#get number of images
num_images = len(xs1)

#shuffle list of images
c = list(zip(xs1, xs2, ys1, ys2, ys3))
random.shuffle(c)
xs1, xs2, ys1, ys2, ys3 = zip(*c)

# Training data
train_xs1 = xs1[:int(len(xs1) * 0.8)]
train_xs2 = xs2[:int(len(xs1) * 0.8)]

train_ys1 = ys1[:int(len(xs1) * 0.8)]
train_ys2 = ys2[:int(len(xs1) * 0.8)]
train_ys3 = ys3[:int(len(xs1) * 0.8)]

# Validation data
val_xs1 = xs1[-int(len(xs1) * 0.2):]
val_xs2 = xs1[-int(len(xs1) * 0.2):]

val_ys1 = ys1[-int(len(xs1) * 0.2):]
val_ys2 = ys2[-int(len(xs1) * 0.2):]
val_ys3 = ys3[-int(len(xs1) * 0.2):]

num_train_images = len(train_xs1)
num_val_images = len(val_xs1)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out1 = []
    x_out2 = []

    y_out1 = []
    y_out2 = []
    y_out3 = []

    for i in range(0, batch_size):
        x_out1.append(scipy.misc.imresize(scipy.misc.imread(train_xs1[(train_batch_pointer + i) % num_train_images])[-150:], [66, 200]) / 255.0)
        x_out2.append(scipy.misc.imresize(scipy.misc.imread(train_xs2[(train_batch_pointer + i) % num_train_images])[-150:], [66, 200]) / 255.0)
        y_out1.append([train_ys1[(train_batch_pointer + i) % num_train_images]])
        y_out2.append([train_ys2[(train_batch_pointer + i) % num_train_images]])
        y_out3.append([train_ys3[(train_batch_pointer + i) % num_train_images]])
        
    train_batch_pointer += batch_size
    return x_out1, x_out2, y_out1, y_out2, y_out3

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out1 = []
    x_out2 = []

    y_out1 = []
    y_out2 = []
    y_out3 = []
 
    for i in range(0, batch_size):
        x_out1.append(scipy.misc.imresize(scipy.misc.imread(val_xs1[(val_batch_pointer + i) % num_val_images])[-150:], [66, 200]) / 255.0)
        x_out2.append(scipy.misc.imresize(scipy.misc.imread(val_xs2[(val_batch_pointer + i) % num_val_images])[-150:], [66, 200]) / 255.0)
        y_out1.append([val_ys1[(val_batch_pointer + i) % num_val_images]])
        y_out2.append([val_ys2[(val_batch_pointer + i) % num_val_images]])
        y_out3.append([val_ys3[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out1, x_out2, y_out1, y_out2, y_out3
