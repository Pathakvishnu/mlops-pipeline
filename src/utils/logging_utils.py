import logging, sys
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name) # gets a logger object for the specific name
    # for instance it helps distiguish beteween logs from multiple file

    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout) # to output to console or terminal
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger
