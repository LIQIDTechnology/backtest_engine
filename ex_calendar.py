import datetime as dt


class Calendar(object):
    def __init__(self, mic_str: str = None):
        self.mic_ls = mic_str

    def bday_add(self, date: dt.date, days: int) -> dt.date:
        new_date = date + dt.timedelta(days=days)
        while 5 <= new_date.weekday() <= 6:
            new_date = new_date + dt.timedelta(days=days)
        return new_date

    def bday_range(self, start: dt, end: dt) -> list:
        numdays = end-start
        start.weekday()
        bday_range = [start + dt.timedelta(days=k) for k in range(0, (numdays.days + 1)) if (start + dt.timedelta(days=k)).weekday() < 5]
        return bday_range

    def __repr__(self):
        return self.mic_ls
