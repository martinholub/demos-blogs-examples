import sys
from datetime import datetime
import logging
from inspect import getmodule

def _L(skip=0):
    '''Shorthand to get logger from some parent frame

    Parmeters:
    -----------
    skip = 0: -> calling function
    skip = 1: -> module importing calling function
    '''
    return logging.getLogger(caller_name(skip + 1))

def caller_name(skip=2):
    """Get a name of a caller module

       `skip` specifies how many levels of stack to skip while getting caller
       name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.

       An empty string is returned if skipped levels exceed stack height

    References:
    --------
      https://gist.github.com/techtonik/2151727
    """
    def stack_(frame):
        framelist = []
        while frame:
            framelist.append(frame)
            frame = frame.f_back
        return framelist

    stack = stack_(sys._getframe(1))
    start = 0 + skip
    if len(stack) < start + 1:
        return ''
    parentframe = stack[start]
    module = getmodule(parentframe)

    if module:
        ret_name = module.__name__
    else:
        ret_name = __name__

    return ret_name

def initialize_logger():
    """Initialize logger

    Creates stream and file handler with different levels

    Returns:
    -----------
    logger: instance of Logger class

    Notes:
    -------------
    simplified setup: |
    	# logging.basicConfig(format='%(asctime)s:%(name)s:%(levelname)s:	 %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG, handlers=[logging.FileHandler("fname"), logging.StreamHandler()])
    """
    # Clean up
    logging.shutdown()
    # Log to file and to stdout
    fname = \
        "logs/aigym_{}.log".format(datetime.now().strftime('%Y%m%d-%H%M%S'))
    # Get logger
    logger = _L(skip =1)

    if has_handlers(logger): # Avoid multiplicative handlers
        teardown_logger(logger)

    logger_fmt = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s')
    logger.setLevel(logging.DEBUG) # set loggers level to the lowest one

    ## file, debug level
    try:
        logger_h1 = logging.FileHandler(fname, mode = "w", encoding = "utf8")
    except FileNotFoundError as e:
        create_logs_folder()
        logger_h1 = logging.FileHandler(fname, mode = "w", encoding = "utf8")

    logger_h1.setLevel(logging.DEBUG)
    logger_h1.setFormatter(logger_fmt)

    ## stdout, info level
    logger_h2 = logging.StreamHandler(sys.stdout)
    logger_h2.setLevel(logging.INFO)
    logger_h2.setFormatter(logger_fmt)

    logger.addHandler(logger_h1)
    logger.addHandler(logger_h2)

    return logger

def teardown_logger(logger):
    """Flushes and closes all handlers of logger.

    Prameters:
    -----------
    logger: instance of Logger class
    """
    while logger.handlers:
        for h in logger.handlers:
            logger.removeHandler(h)
            h.flush()
            h.close()
    logging.shutdown()

def has_handlers(name):
    """ Check if logger has any handlers associated with it

    Parameters
    ---------
    name: logging.Logger or str, logger instance or its name
    """
    ret_val = 0
    if type(name) == logging.Logger:
        hndls = name.handlers
    else:
        try:
            my_logger = logging.Logger.manager.loggerDict[name]
            hndls = my_logger.handlers
        except KeyError as e:
            logger.debug("No logger named {}".format(name))
            hndls = []

    if len(hndls) > 0:
        ret_val = 1

    return ret_val

def create_logs_folder(parent = ".", dirs = ["logs"]):
    """Creates folder structure to store simulation results (data, plots, logs)
    """
    # if not os.path.isdir(parent):
    #     os.mkdir(parent)
    for di in dirs:
        dir_path = os.path.join(parent, di)
        if os.path.isdir(dir_path):
            continue
        else:
            logger.info("Creating {} in {}".format(di, parent))
            os.mkdir(dir_path)
