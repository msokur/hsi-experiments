import itertools as it
import os
import glob
import numpy as np
import inspect

from configuration.keys import TelegramKeys as TGK


class Telegram:
    def __init__(self, tg_config: dict, mode: str):
        self.tg = tg_config
        self.mode = mode

    def send_tg_message(self, message):
        import telegram_send
        if self.tg[TGK.SENDING]:
            if self.mode == "CLUSTER":
                message = "CLUSTER " + message
            try:
                message = f"{self.tg[TGK.USER]}, " + message
                telegram_send.send(messages=[message], conf=self.tg[TGK.FILE])
            except Exception as e:
                print("Some problems with telegram! Messages could not be delivered")
                print(e)

    def send_tg_message_history(self, log_dir, history):
        if history is not None:
            self.send_tg_message(
                f"{self.tg[TGK.USER]}, training {log_dir} has finished after {len(history.history['loss'])} epochs")
        else:
            self.send_tg_message(f"{self.tg[TGK.USER]}, training {log_dir} has finished")


def glob_multiple_file_types(path, *patterns):
    return list(it.chain.from_iterable(glob.iglob(os.path.join(path, pattern)) for pattern in patterns))


def round_to_the_nearest_even_int(number, nearest_int):  # =config.WRITE_CHECKPOINT_EVERY_Xth_STEP):
    return int(np.round(number / nearest_int) * nearest_int)


def print_function_signature(func, name, printing=False):
    if printing:
        print(f'--------------------method {name} params----------------------')
        signature = inspect.signature(func)
        for param in signature.parameters.values():
            print(param)
        print('------------------------------------------------')
