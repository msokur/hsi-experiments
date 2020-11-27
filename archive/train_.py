from data_loader import *
import tensorflow as tf 
import config
import datetime
from data_loader import get_data_for_showing
from model import get_model


'''-------DATASET---------'''
train, test = get_data()

train_dataset = tf.data.Dataset.from_tensor_slices((train[:, :-1], train[:, -1])).batch(config.BATCH_SIZE)
test_dataset =  tf.data.Dataset.from_tensor_slices((test[:, :-1], test[:, -1])).batch(config.BATCH_SIZE)

print(next(iter(train_dataset)), next(iter(test_dataset)))

'''-------MODEL---------'''

model = get_model()

'''-------LOSS, ACCURACY---------'''

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')


train_sensitivity_0 = tf.keras.metrics.Recall(class_id = 0, name='train_sensivity_recall_0')
train_sensitivity_1 = tf.keras.metrics.Recall(class_id = 1, name='train_sensivity_recall_1')
train_sensitivity_2 = tf.keras.metrics.Recall(class_id = 2, name='train_sensivity_recall_2')

test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
test_sensitivity_0 = tf.keras.metrics.Recall(class_id = 0, name='test_sensivity_recall_0')
test_sensitivity_1 = tf.keras.metrics.Recall(class_id = 1, name='test_sensivity_recall_1')
test_sensitivity_2 = tf.keras.metrics.Recall(class_id = 2, name='test_sensivity_recall_2')


def train_step(model, optimizer, x_train, y_train):
  with tf.GradientTape() as tape:
    predictions = model(x_train, training=True)
    loss = loss_object(y_train, predictions)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  train_loss(loss)
  train_accuracy(y_train, predictions)

  y_train_ = tf.one_hot(tf.cast(y_train, dtype=tf.uint8), depth = 3)
  predictions_ = tf.nn.softmax(predictions)


  train_sensitivity_0(y_train_, predictions_)
  train_sensitivity_1(y_train_, predictions_)
  train_sensitivity_2(y_train_, predictions_)

def test_step(model, x_test, y_test):
  predictions = model(x_test)
  loss = loss_object(y_test, predictions)

  test_loss(loss)
  test_accuracy(y_test, predictions)

  y_test_ = tf.one_hot(tf.cast(y_test, dtype=tf.uint8), depth = 3)
  predictions_ = tf.nn.softmax(predictions)

  test_sensitivity_0(y_test_, predictions_)
  test_sensitivity_1(y_test_, predictions_)
  test_sensitivity_2(y_test_, predictions_)


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
#test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_log_dir = 'logs/gradient_tape/super_small_conv1d' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/super_small_conv1d' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

gt_image, spectrum_data, indexes = get_data_for_showing()


def draw_predictions_on_images(predictions):
    #images = np.round(images * 255).astype(np.uint8)
    image = gt_image.copy()#cv2.imread('2019_07_15_11_33_28_SpecCube.dat_Mask JW Kolo2.png')
    
    for counter, value in tqdm(enumerate(predictions)):
      key = int(value)

      if key == 0:
        image[indexes[counter, 0], indexes[counter, 1]] = [255, 0, 0]
      elif key == 1:
        image[indexes[counter, 0], indexes[counter, 1]] = [0, 0, 255]
      else:
        image[indexes[counter, 0], indexes[counter, 1]] = [255, 255, 0]

    image = np.concatenate((image, gt_image), axis=0)
    cv2.imwrite('test.png', image)
    return image

EPOCHS = 5000

for epoch in range(EPOCHS):
  for (x_train, y_train) in train_dataset:
    train_step(model, optimizer, x_train, y_train)
  with train_summary_writer.as_default():
    tf.summary.scalar('loss', train_loss.result(), step=epoch)
    tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    tf.summary.scalar('sensitivity_gesund', train_sensitivity_0.result(), step=epoch)
    tf.summary.scalar('sensitivity_ill', train_sensitivity_1.result(), step=epoch)
    tf.summary.scalar('sensitivity_not_certain', train_sensitivity_2.result(), step=epoch)

    model.save('model.')
    
    if epoch % 50 == 0:
      predictions = tf.argmax(tf.nn.softmax(model(spectrum_data[indexes[:, 1], indexes[:, 0]])), axis=1)
      image = np.array(tf.py_function(draw_predictions_on_images, [predictions], [tf.uint8]))
      tf.summary.image('gt', image[..., ::-1], step=epoch)

  for (x_test, y_test) in test_dataset:
    test_step(model, x_test, y_test)
  with test_summary_writer.as_default():
    tf.summary.scalar('loss', test_loss.result(), step=epoch)
    tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
    
    tf.summary.scalar('sensitivity_gesund', test_sensitivity_0.result(), step=epoch)
    tf.summary.scalar('sensitivity_ill', test_sensitivity_1.result(), step=epoch)
    tf.summary.scalar('sensitivity_not_certain', test_sensitivity_2.result(), step=epoch)

  
  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(), 
                         train_accuracy.result()*100,
                         test_loss.result(), 
                         test_accuracy.result()*100))

  # Reset metrics every epoch
  train_loss.reset_states()
  test_loss.reset_states()
  train_accuracy.reset_states()
  test_accuracy.reset_states()