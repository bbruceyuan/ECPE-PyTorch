__all__ = ["LOG_CONFIG"]

LOG_CONFIG = {
    'disable_existing_loggers': False,
    'version': 1,
    'formatters': {
        'short': {
            'format': '%(asctime)s %(levelname)s %(name)s: %(message)s'
        },
        # this is the concrete info of log format
        'concrete': {
            'format': '%(asctime)s %(levelname)s %(funcName)s %(lineno)d %(name)s: ' \
                      '%(message)s %(funcName)s ' \
                      '%(pathname)s'
        }
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'formatter': 'short',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['console'],
            'level': 'DEBUG',
        },
        'plugins': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False
        }
    },
}
