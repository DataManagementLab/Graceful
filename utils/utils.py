from datetime import datetime


def t_print(text: str, flush: bool = False):
    """
    Prepend text print with timestamp
    :param text:
    :return:
    """
    print(
        f'{str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}: {text}', flush=flush)
