from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import gc

def train_model(X_train, y_train, X_val, y_val, model, model_name, epochs, batch_size, save_path, save_best_only=False):
    def lr_scheduler(epoch, lr, warmup_epochs=epochs//5, decay_epochs=epochs*2//3, initial_lr=1e-6, base_lr=1e-3, min_lr=5e-5):
        if epoch <= warmup_epochs:
            pct = epoch / warmup_epochs
            return ((base_lr - initial_lr) * pct) + initial_lr

        if epoch > warmup_epochs and epoch < warmup_epochs+decay_epochs:
            pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
            return ((base_lr - min_lr) * pct) + min_lr

        return min_lr

    model.compile(optimizer=Adam(0.001), loss='mse')

    checkpoint = ModelCheckpoint(save_path + f'{model_name}.h5', 
                                        monitor='val_loss',
                                        save_best_only=save_best_only)
    
    learningrate_scheduler = LearningRateScheduler(lr_scheduler)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=epochs//10, verbose=0,)
    
    model.fit(X_train, y_train, 
            validation_data = (X_val, y_val), 
            batch_size = batch_size,
            epochs = epochs,
            callbacks=[
                       checkpoint,  
                       learningrate_scheduler,
                       early_stop
                       ]
            )
    
    del model
    gc.collect()
    K.clear_session()

def build_bidirectional_lstm(input_shape, n_units, middle_dense_dim=None, dropout=None, kernel_initializer='glorot_uniform'):
    model = Sequential()
    
    for i, n_unit in enumerate(n_units):
        if len(n_units) == 1:
            model.add(
                  L.Bidirectional(L.LSTM(n_unit, 
                              kernel_initializer=kernel_initializer
                              ),
                              input_shape=input_shape,)
            )
            
        else:
            if i == 0:
                model.add(
                    L.Bidirectional(L.LSTM(n_unit, 
                                return_sequences = True,
                                kernel_initializer=kernel_initializer
                                ),
                                input_shape=input_shape,)
                )
            elif i == len(n_units) - 1:
                model.add(
                    L.Bidirectional(L.LSTM(n_unit, 
                                return_sequences = False,
                                kernel_initializer=kernel_initializer
                                ))
                )
            else:
                model.add(
                    L.Bidirectional(L.LSTM(n_unit, 
                                return_sequences = True,
                                kernel_initializer=kernel_initializer
                                ))
                )
    if middle_dense_dim:
        model.add(
            L.Dense(middle_dense_dim,
                    activation = 'relu',
                    kernel_initializer=kernel_initializer)
        )

    if dropout:
        model.add(
            L.Dropout(dropout)
        )

    model.add(
          L.Dense(1, activation='linear', kernel_initializer=kernel_initializer)
      )

    model.summary()
    
    return model
