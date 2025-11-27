"""Training script: loads numpy arrays and trains model from 2_model.py"""
import os, numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import _0_config as cfg
from _2_model import build_convlstm_model

def load_data():
    train_X = np.load(os.path.join(cfg.OUT_NP_DIR, 'train_X.npy'))
    train_Y = np.load(os.path.join(cfg.OUT_NP_DIR, 'train_Y.npy'))
    val_X = np.load(os.path.join(cfg.OUT_NP_DIR, 'val_X.npy'))
    val_Y = np.load(os.path.join(cfg.OUT_NP_DIR, 'val_Y.npy'))
    
    
    train_X = train_X[..., None]
    
    train_Y = np.squeeze(train_Y)
    train_Y = np.expand_dims(train_Y, axis=-1)
    
    val_X = val_X[..., None]
    
    val_Y = np.squeeze(val_Y)
    val_Y = np.expand_dims(val_Y, axis=-1)
    
    return train_X, train_Y, val_X, val_Y

def main():
    train_X, train_Y, val_X, val_Y = load_data()
    print('Data shapes:', train_X.shape, train_Y.shape, val_X.shape, val_Y.shape)
    model = build_convlstm_model(in_seq=cfg.IN_SEQ, h=cfg.IMG_SHAPE[0], w=cfg.IMG_SHAPE[1], filters=32)
    ckpt_path = os.path.join(cfg.OUT_DIR, 'best_model.keras')
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    callbacks = [
        ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss'),
        EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss')
    ]
    history = model.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=50, batch_size=8, callbacks=callbacks)
    model.save(os.path.join(cfg.OUT_DIR, 'final_model.keras'))
    print('Saved model to', cfg.OUT_DIR)

if __name__ == '__main__':
    main()