import datetime as dt

class Calendar(object):

    def __init__(self, mic_str: str = None):
        self.mic_ls = mic_str

    def __iter__(self):
        for exchange in self.mic_ls:
            yield exchange
