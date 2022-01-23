import telegram_send
import config

def send_tg_message(message):
    if config.TELEGRAM_SENDING:
        if config.MODE == 'SERVER':
                message = 'SERVER ' + message
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