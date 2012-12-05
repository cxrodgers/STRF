import numpy as np
import io
from io import STRFlabFileSchema

def allclose_2d(a):
    """Returns True if entries in `a` are close to equal.
    
    a : iterable, convertable to array
    
    If the entries are arrays of different size, returns False.
    If len(a) == 1, returns True.
    """
    a = np.asarray(a)
    if a.ndim == 0:
        raise ValueError("input to allclose_2d cannot be 0d")
    return np.all([np.allclose(a[0], aa) for aa in a[1:]])

def concatenate_and_reshape_timefreq(timefreq_list, n_delays, 
    blanking_value=-np.inf):
    """Concatenate and reshape spectrograms for STRF estimation.
    
    Returned value is one large spectogram of shape (n_freqs*n_delays, 
    total_time). Each column contains the current value and the previous
    n_delays.    
    
    Arugments:
    timefreq_list : list of 2d arrays
        Each array has shape (n_freqs, n_timepoints)
    
    n_delays : int
        Number of blank timepoints to insert between spectrograms.
    
    blanking_value : float
        What value to insert at the blank timepoints.
    
    Puts (n_freqs, n_delays) blanking_values before each stimulus.
    Then reshapes each column to include delays:
        column_n = concatenated_specgram[:, n:n-n_delays:-1].flatten()
    
    The original slice corresponding to column n:
        concatenated_specgram[:, n:n-n_delays:-1]
    can be recovered as:
        reshaped_specgram[:, n].reshape(specgram.shape[0], n_delays)
    
    There are n_delays blanks in between each stimuli, but the total length
    of the returned value is the sum of the total length of the provided
    stimuli because those blanks are folded into the features. That is, the
    first entry contains the first time sample and the rest blanks; and
    the last sample contains the last n_delays samples.
    """   
    # put blanks in front of each stimulus and concatenate
    concatenated_specgram_list = []
    for specgram in timefreq_list:
        # first prepend blanks
        specgram_with_prepended_blanks = np.concatenate([
            blanking_value * np.ones((specgram.shape[0], n_delays)), 
            specgram], axis=1)
        
        # now reshape and include delays in each feature
        reshaped_specgram_list = []
        for n in range(n_delays, specgram_with_prepended_blanks.shape[1]):
            reshaped_specgram_list.append(
                specgram_with_prepended_blanks[:, n:n-n_delays:-1].flatten())
        reshaped_specgram = np.transpose(np.array(reshaped_specgram_list))
        
        concatenated_specgram_list.append(reshaped_specgram)
    concatenated_specgram = np.concatenate(concatenated_specgram_list, axis=1)
    
    return concatenated_specgram  


class Experiment:
    """Object encapsulating STRF estimation for a specific dataset"""
    def __init__(self, path=None, file_schema=None):
        """Create a new object to estimate a STRF from a dataset.
        
        """
        # Location of data
        self.path = path
        if file_schema is None:
            self.file_schema = STRFlabFileSchema(self.path)
        
        # How to read timefreq files
        self.timefreq_file_reader = io.read_timefreq_from_matfile

    def read_timefreq(self, label):
        filename = self.file_schema.timefreq_filename[label]        
        return self.timefreq_file_reader(filename)
    
    def read_all_timefreq(self, store_intermediates=True):
        """Read timefreq from disk. Store and return.
        
        Reads all timefreq from self.file_schema.
        Each consists of Pxx, freqs, t.
        
        If the freqs is the same for all, then stores in self.freqs.
        Otherwise, self.freqs is None.
        Same for t.
        
        Returns:
            List of Pxx, list of freqs, list of t
        """
        # Load all
        Pxx_l, freqs_l, t_l = zip(*[self.read_timefreq(label)   
            for label in self.file_schema.timefreq_file_labels])
        
        # Optionally store
        if store_intermediates:
            self.timefreq_list = Pxx_l
            self.freqs_l = freqs_l
            self.t_l = t_l
    
        # Test for freqs consistency
        self.freqs = None
        if allclose_2d(freqs_l):
            self.freqs = np.mean(freqs_l, axis=0)

        # Test for t consistency
        self.t = None
        if allclose_2d(t_l):
            self.t = np.mean(t_l, axis=0)
        
        return Pxx_l, freqs_l, t_l
    
    def read_response(self, label):
        folded = io.read_single_stimulus(self.file_schema.spike_path, label)
        return folded
    
    def read_all_responses(self):
        """Reads all response files and stores in self.response_l"""
        # Read in all spikes
        dfolded = io.read_directory(self.file_schema.spike_path)
        
        # Order by label
        response_l = []
        for label in self.file_schema.spike_file_labels:
            response_l.append(dfolded[label])
        
        self.response_l = response_l
        return response_l
    
    def compute_binned_responses(self):
        """Bins the stored responses in the same way as the stimuli"""
        assert len(self.t_l) == len(self.response_l)
        
        # Iterate over stimuli
        for folded, t_stim in zip(self.response_l, self.t_l):
            # Bin each, using the same number of bins as in t
            binned = kkpandas.Binned.from_folded(folded, bins=len(t_stim))
            
            # Check that the starts and stops line up
            1/0

    def compute_full_stimulus_matrix(self, n_delays=3, timefreq_list=None,
        blanking_value=-np.inf):
        """Given a list of spectrograms, returns the full stimulus matrix."""
        # Determine what list to operate on
        if timefreq_list is None:
            timefreq_list = self.timefreq_list
        if timefreq_list is None:
            timefreq_list = self.read_all_timefreq()[0]
        if timefreq_list is None:
            raise ValueError("cannot determine timefreq lists")

        # Concatenate and reshape
        self.full_stimulus_matrix = concatenate_and_reshape_timefreq(
            timefreq_list, 
            n_delays=n_delays, blanking_value=blanking_value)

        # Write out

        return self.full_stimulus_matrix