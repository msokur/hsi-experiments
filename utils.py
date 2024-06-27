import itertools as it
import os
import glob
import numpy as np
import inspect
import psutil
import asyncio
import re

from configuration.keys import TelegramKeys as TGK


class Telegram:
    def __init__(self, tg_config: dict, mode: str):
        self.tg_config = tg_config
        self.mode = mode

    def send_tg_message(self, message):
        if self.tg_config[TGK.SENDING]:
            if self.mode == "CLUSTER":
                message = "CLUSTER " + message
            try:
                import telegram_send
                import asyncio
                import platform
                if platform.system() == 'Windows':
                    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

                message = f"{self.tg_config[TGK.USER]}, " + message
                asyncio.run(telegram_send.send(messages=[message], conf=self.tg_config[TGK.FILE]))
            except Exception as e:
                print("Some problems with telegram! Messages could not be delivered")
                print(e)

    def send_tg_message_history(self, log_dir, history):
        if history is not None:
            self.send_tg_message(
                f"{self.tg_config[TGK.USER]}, training {log_dir} "
                f"has finished after {len(history.history['loss'])} epochs")
        else:
            self.send_tg_message(f"{self.tg_config[TGK.USER]}, training {log_dir} has finished")


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


def get_used_memory(process_id: int, unit: str = "GB") -> str:
    unit = unit.upper()
    exponent = 0

    if unit == "KB":
        exponent = 1
    elif unit == "MB":
        exponent = 2
    elif unit == "GB":
        exponent = 3
    else:
        unit = "BYTES"

    process = psutil.Process(process_id)
    memory = process.memory_info().rss
    memory /= 1024 ** exponent
    return f"{round(memory, 3)} {unit}"


def alphanum_key(s: str):
    file_name = os.path.basename(s)
    return [_try_int(c) for c in re.split(pattern="([0-9]+)", string=file_name)]


def _try_int(s):
    try:
        return int(s)
    except ValueError:
        return s
