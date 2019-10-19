import numpy as np
import os.path
import tensorflow as tf
from tensorflow import keras
def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(23,)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(23)
  ])
 
  #optimizer = tf.train.RMSPropOptimizer(0.001)
  optimizer = tf.train.AdamOptimizer()
 
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

def main():
    original_data=np.loadtxt("./log/state_transition.log",delimiter=',')
    data=original_data[:900000]
    train_input=data[::2]
    train_output=data[1::2]
    print(data.shape)
    print(train_input.shape,train_output.shape)

    order = np.argsort(np.random.random(train_output.shape[0]))
    train_input=train_input[order]
    train_output=train_output[order]
    print(train_input.shape,train_output.shape)
    test_data=original_data[900000:]
    test_input=test_data[::2]
    test_output=test_data[1::2]
    print(test_input.shape,test_output.shape)

    model = build_model()
    model.summary()

    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs):
            if epoch % 100 == 0: print('')
            print('.', end='')
    
    EPOCHS = 2000#500
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)
    # Store training stats
    history = model.fit(train_input, train_output, epochs=EPOCHS,
                        validation_split=0.2, verbose=0,
                        callbacks=[early_stop,PrintDot()])

    import matplotlib.pyplot as plt
    
    
    def plot_history(history):
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error ')
        plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
                label='Train Loss')
        plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
                label = 'Val loss')
        plt.legend()
        plt.ylim([0,5])
        plt.show()

    [loss, mae] = model.evaluate(train_input, train_output, verbose=0)
    
    print("Testing set Mean Abs Error: {:7.2f}".format(mae))

    test_predictions = model.predict(test_input)
    print(np.mean(abs(test_predictions-test_output)))
    #print(sum(abs(test_predictions-test_output))/len(test_input))
    plot_history(history)
    f_model = './model'
    json_string=model.to_json()
    open(os.path.join(f_model,'battle_state_model_v2.json'),'w').write(json_string)
    yaml_string=model.to_yaml()
    open(os.path.join(f_model,'battle_state_model_v2.yaml'),'w').write(yaml_string)
    model.save_weights(os.path.join(f_model,'battle_state_model_v2.hdf5'))

if __name__ == "__main__":
    main()