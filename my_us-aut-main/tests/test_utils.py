import os
import sys
from unittest import TestCase

import numpy as np
import pandas as pd
import logging

sys.path.insert(0, os.path.abspath(".."))

from src.customer_analysis.common import logg



class Testingest(TestCase):
    def test_is_df(self):
        logger=logg()
        assert isinstance(logger, logging.RootLogger)
