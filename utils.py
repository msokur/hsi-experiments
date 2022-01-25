import telegram_send
import config
import itertools as it, glob
import os
import glob
import numpy as np

def send_tg_message(message):
    if config.TELEGRAM_SENDING:
        if config.MODE == 'CLUSTER':
                message = 'CLUSTER ' + message
        try:
            telegram_send.send(messages=[message], conf='~/hsi-experiments/tg.config')
        except Exception as e:
            print('Some problems with telegram! Messages could not be delivered')
            print(e)

def send_tg_message_history(log_dir, history):
    if history is not None:
        send_tg_message(f'Mariia, training {log_dir} has finished after {len(history.history["loss"])} epochs')
    else:
        send_tg_message(f'Mariia, training {log_dir} has finished')
        
def multiple_file_types(path, *patterns):
    return list(it.chain.from_iterable(glob.iglob(os.path.join(path, pattern)) for pattern in patterns))

def round_to_the_nearest_int(number):
    return int(np.round(number / 2) * 2)