from typing import List, Tuple
import logging
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)


class PairMetrics(object):
    def __init__(self):
        super(PairMetrics, self).__init__()
