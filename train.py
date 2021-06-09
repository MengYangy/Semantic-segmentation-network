from Nets.MyModels import Feature_Extract_Net
from utili.LoadData import My_LoadData
import tensorflow as tf
from utili.My_LR import Mylearning_rate
import matplotlib.pyplot as plt
import numpy as np
from utili.myEvalue import PA, precision,recall,f1_score,iou,aiou

def create_data(BATCH_SIZE = 2):
    imgs_dir = 'C:/Users/MYY/Desktop/MyCode/data/trains/'
    train_img_dir = 'C:/Users/MYY/Desktop/MyCode/data/trains/'
    train_lab_dir = 'C:/Users/MYY/Desktop/MyCode/data/labels/'
    val_img_dir = 'C:/Users/MYY/Desktop/MyCode/data/trains/'
    val_lab_dir = 'C:/Users/MYY/Desktop/MyCode/data/labels/'


    genarate = My_LoadData()
    train_name, val_name = genarate.get_train_val_fileName(imgs_dir)
    train_generate = genarate.generate_train_val_data(img_dir=train_img_dir,
                                                      lab_dir=train_lab_dir,
                                                      batch_size=BATCH_SIZE,
                                                      data_name=train_name)

    val_generate = genarate.generate_train_val_data(img_dir=val_img_dir,
                                                    lab_dir=val_lab_dir,
                                                    batch_size=BATCH_SIZE,
                                                    data_name=val_name)
    return train_generate, val_generate, len(train_name), len(val_name)

def lrfn(epoch, init_lr=0.05, attenuation_rate=0.5, attenuation_step=5):

    lr = init_lr
    lr = lr * attenuation_rate**(epoch//attenuation_step)
    return lr


if __name__ == '__main__':

    EPOCHS = 50
    L_EP = 0
    BATCH_SIZE = 2
    train_generate, val_generate, train_num, val_num = create_data(BATCH_SIZE=BATCH_SIZE)
    STEPS_PER_EPOCH = train_num // BATCH_SIZE
    val_STEPS_PER_EPOCH = val_num // BATCH_SIZE

    learning_rate = Mylearning_rate()
    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        learning_rate.fixed_step_attenuation, verbose=True)

    models = Feature_Extract_Net()

    model = models.feature_Extract(name='res50')
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['acc', PA, precision,recall,f1_score,iou,aiou])
    # precision,recall,f1_score,iou,aiou

    history = model.fit(train_generate,
                        epochs=EPOCHS,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        initial_epoch=L_EP,
                        validation_data=val_generate,
                        validation_steps=val_STEPS_PER_EPOCH,
                        callbacks=[lr_callback]  # DisplayCallback(),
                        #                     callbacks=[reduce_lr]
                        )

    fig1 = plt.figure(num='Acc/Loss', figsize=(8, 8), dpi=75, facecolor='#FFFFFF', edgecolor='#0000FF')
    plt.plot(np.arange(L_EP, EPOCHS), history.history["loss"], label="train_loss")
    plt.plot(np.arange(L_EP, EPOCHS), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(L_EP, EPOCHS), history.history["acc"], label="train_acc")
    plt.plot(np.arange(L_EP, EPOCHS), history.history["val_acc"], label="val_acc")
    plt.legend(["train_loss", "val_loss", "train_acc", "val_acc"])
    plt.savefig("Acc_loss.jpg")
    plt.show()