import datetime


class CustomLogger:
    def __init__(self, location):
        self.location = location

    def info(self, msg):
        with open(self.location, "a+") as f:
            f.write(datetime.datetime.now().strftime("%m_%d_%H_%M_%S"))
            f.write("\t")
            f.write(msg)
            f.write("\n")
