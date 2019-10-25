import logging
import logging.handlers
def get_module_logger(modname):
    #logging.basicConfig(format='%(name)s:%(message)s')
    logger = logging.getLogger(modname)
    logger.propagate=False
    logger.setLevel(logging.INFO)
    #logger.setLevel(logging.WARNING)
    #logger.propagate = False
    streamhandler = logging.StreamHandler()
    #handler2 = logging.handlers.TimedRotatingFileHandler('log/battle.log',when='H',interval=1,backupCount=24)
    formatter = logging.Formatter('%(name)s:%(message)s')
    streamhandler.setFormatter(formatter)
    logger.addHandler(streamhandler)
    #filehandler = logging.FileHandler(filename='log/battle.log')
    # 
    #filehandler.setFormatter(formatter)
    #logger.addHandler(filehandler)
    #logger.addHandler(handler2)
    return logger

def get_state_logger(modname):
    logger = logging.getLogger(modname)
    logger.propagate=False
    logger.setLevel(logging.INFO)
    """
    handler = logging.handlers.RotatingFileHandler(\
        filename='log/state_transition.log',maxBytes=10000000000,\
            backupCount=5,encoding='utf-8'
    )
    """

    handler = logging.FileHandler(filename='log/state_transition.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger

