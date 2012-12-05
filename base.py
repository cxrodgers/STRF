import numpy as np
import io
from io import STRFlabFileSchema

class Experiment:
    """Object encapsulating STRF estimation for a specific dataset"""
    def __init__(self, path=None, file_schema=None):
        """Create a new object to estimate a STRF from a dataset.
        
        """
        # Location of data
        self.path = path
        if file_schema is None:
            self.file_schema = STRFlabFileSchema(self.path)
