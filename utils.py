import telegram_send
import config

def send_tg_message(message):
    if config.MODE == 0:
            message = 'SERVER ' + message
    
    try:    
        telegram_send.send(messages=[message], conf='~/hsi-experiments/tg.config')
    except Exception as e:
        print('Some problems with telegram! Messages could not be delivered')
        print(e)