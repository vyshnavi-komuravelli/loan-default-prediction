import logging

def logg():
    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG)
    format=logging.Formatter("%(levelname)s:%(message)s")
    file_handler=logging.StreamHandler()
    logger.addHandler(file_handler)
    return logger

