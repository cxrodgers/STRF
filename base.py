import numpy as np
import io
from io import STRFlabFileSchema
import kkpandas, pandas



# Define the algorithms
def fit_lstsq(A, b):
    return np.linalg.lstsq(A, b)[0]

def fit_ridge(A, b, alpha=None, ATA=None):
    # Calculate ATA if necessary
    if ATA is None:
        ATA = np.dot(A.T, A)
    
    if alpha is None:
        # No ridge, just pinv
        to_invert = ATA
    else:
        # Ridge at alpha
        to_invert = ATA + alpha * (len(ATA) ** 2) * np.eye(len(ATA))
    return reduce(np.dot, (np.linalg.inv(to_invert), A.T, b))

def fit_STA(A, b):
    return np.dot(A.T, b) / b.sum()

def check_fit(A, b, X_fit=None, b_pred=None, scale_to_fit=False):
    if b_pred is None:
        b_pred = np.dot(A, X_fit)
    if scale_to_fit:
        b_pred = b_pred * np.sqrt(b.var() / b_pred.var())
    err = b - b_pred
    
    # Calculate xcorr
    bx = ((b - b.mean()) / b.std()).flatten()
    b_predx = ((b_pred - b_pred.mean()) / b_pred.std()).flatten()
    xcorr = np.inner(bx, b_predx) / float(len(bx))
    return err.var(), err.mean(), b_pred.var(), err.var() / b.var(), xcorr

def jackknife_over_alpha(A, b, alphas, n_jacks=5, meth=fit_ridge,
    keep_fits=False, warn=True):
    # warn
    if warn:
        if not np.allclose(0, A.mean(axis=0)):
            print "warning: A is not zero meaned"
        if not np.allclose(0, b.mean(axis=0)):
            print "warning: b is not zero meaned"
    
    # Set up the analyses
    analyses = [
        ['ridge%010.05f' % alpha, meth, {'alpha': alpha}]
        for alpha in alphas]
    
    # set up the jackknife
    jk_len = len(A) / n_jacks
    jk_starts = np.arange(0, len(A) - jk_len + 1, jk_len)

    # set up jk_results
    jk_metrics = []
    jk_fits = []
    
    # Jack the knifes
    for n, jk_start in enumerate(jk_starts):
        # Set up test and train sets
        jk_idxs = np.arange(jk_start, jk_start + jk_len)
        jk_mask = np.zeros(len(A), dtype=np.bool)
        jk_mask[jk_idxs] = 1
        A_test, A_train = A[jk_mask], A[~jk_mask]
        b_test, b_train = b[jk_mask], b[~jk_mask]

        # Run the analyses
        results = []
        for name, meth, kwargs in analyses:
            X_fit = meth(A_train, b_train, **kwargs)
            evar, ebias, predvar, eratio, xcorr = check_fit(
                A_train, b_train, X_fit)
            evarsc, ebiassc, predvarsc, eratiosc, xcorrsc = check_fit(
                A_test, b_test, X_fit)
            results.append((name, X_fit, evar, ebias, predvar, eratio, xcorr,
                evarsc, ebiassc, predvarsc, eratiosc, xcorrsc))

        # DataFrame it
        metrics = pandas.DataFrame(results, 
            columns=['name', 'fit', 'evar', 'ebias', 'predvar', 'eratio', 'xcorr',
                'evarsc', 'ebiassc', 'predvarsc', 'eratiosc', 'xcorrsc'])
        metrics = metrics.set_index('name')
        fits = metrics.pop('fit')

        #~ # Make predictions
        #~ preds = pandas.Series([np.dot(A_test, fit) for fit in fits], 
            #~ index=fits.index)
        
        # Store
        jk_metrics.append(metrics)
        if keep_fits:
            jk_fits.append(fits)
        #~ jk_preds.append(preds)
    
    return jk_metrics, jk_fits

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
    def __init__(self, path=None, file_schema=None, timefreq_path=None, **file_schema_kwargs):
        """Create a new object to estimate a STRF from a dataset.
        
        There are many computation steps which must be done in order.
        Here is a full pipeline illustrating its use:
        # Get the files
        expt_path = expt_path_l[0]
        expt = STRF.base.Experiment(expt_path)
        expt.file_schema.timefreq_path = timefreq_dir
        expt.file_schema.populate()

        # Load the timefreq and concatenate
        expt.read_all_timefreq()
        expt.compute_full_stimulus_matrix()

        # Load responses and bin
        expt.read_all_responses()
        expt.compute_binned_responses()

        # Grab the stimulus and responses
        fsm = expt.compute_full_stimulus_matrix()
        frm = expt.compute_full_response_matrix()        
        """
        # Location of data
        self.path = path
        if file_schema is None:
            self.file_schema = STRFlabFileSchema(self.path, **file_schema_kwargs)
            
            # Hack to make it load the timefreq files
            self.file_schema.timefreq_path = timefreq_path
            self.file_schema.populate()
        
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
        # Read in all spikes, recentering
        dfolded = io.read_directory(self.file_schema.spike_path,
            subtract_off_center=True)
        
        # Order by label
        response_l = []
        for label in self.file_schema.spike_file_labels:
            response_l.append(dfolded[label])
        
        self.response_l = response_l
        return response_l
    
    def compute_binned_responses(self, dilation_before_binning=.99663):
        """Bins the stored responses in the same way as the stimuli.
        
        The bins are inferred from the binwidth of the timefreq, as stored
        in self.t_l, independently for each stimulus.
        
        Optionally, a dilation is applied to these bins to convert them into
        the neural timebase.
        
        Finally, the values in self.response_l are binned and stored in 
        self.binned_response_l
        
        I also store self.trials_l to identify how many repetitions of each
        timepoint occurred.
        """
        self.binned_response_l = []
        self.trials_l = []
        
        # Iterate over stimuli
        for folded, t_stim in zip(self.response_l, self.t_l):
            # Get bins from t_stim by recovering original edges
            t_stim_width = np.mean(np.diff(t_stim))
            edges = np.linspace(0, len(t_stim) * t_stim_width, len(t_stim) + 1)
            
            # Optionally apply a kkpandas dilation
            # Spike times are always shorter than behavior times
            edges = edges * dilation_before_binning
            
            # Bin each, using the same number of bins as in t
            binned = kkpandas.Binned.from_folded(folded, bins=edges)
            
            # Save the results
            self.binned_response_l.append(binned.rate.values.flatten())
            self.trials_l.append(binned.trials.values.flatten())

    def compute_concatenated_stimuli(self):
        """Returns concatenated spectrograms as (N_freqs, N_timepoints).
        
        This is really only for visualization, not computation, because
        it doesn't include the delays.
        """
        return np.concatenate(self.timefreq_list, axis=1)
    
    def compute_concatenated_responses(self):
        """Returns a 1d array of concatenated binned responses"""
        return np.concatenate(self.binned_response_l)

    def compute_full_stimulus_matrix(self, n_delays=3, timefreq_list=None,
        blanking_value=-np.inf):
        """Given a list of spectrograms, returns the full stimulus matrix.
        
        See concatenate_and_reshape_timefreq for the implementation details.
        
        This function actually returns a transposed version, more suitable
        for fitting. The shape is (N_timepoints, N_freqs * N_delays),
        ie, (N_constraints, N_inputs)
        """
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
            n_delays=n_delays, blanking_value=blanking_value).T

        # Write out

        return self.full_stimulus_matrix
    
    def compute_full_response_matrix(self):
        """Returns a response matrix, suitable for fitting.
        
        Returned array has shape (N_timepoints, 1)
        """
        self.full_response_matrix = \
            self.compute_concatenated_responses()[:, None]
        return self.full_response_matrix
    


def clean_up_stimulus(whole_stimulus, silence_value='min_row', z_score=True):
    """Replaces non-finite values with silence_value, and z-scores by row.
    
    silence_value == 'min_row' : minimum value in the row
    silence_value == 'min_whole' : minimum value in the whole stimulus
    silence_value == 'mean_row', 'mean_whole' : mean
    silence_value == 'median_row', 'median_whole' : median
    
    You might want to check a histogram of the returned values and pick what
    looks best. On the one hand, silence is best represented by minimal
    power. On the other hand, this distorts the histogram due to silence
    in the actual signals, and/or the blanking periods. It also means that
    this silence will play a part in the linear fit.
    
    Most models treat the rows independently, so if you set the silence
    row-independently, it will mesh nicely. On the other hand, why should
    silence in one frequency band be treated differently from others?
    """
    #~ if np.any(np.isnan(whole_stimulus)):
        #~ print "warning: what to do with NaNs?"
    
    
    if silence_value is 'min_row':
        cleaned_stimulus_a = whole_stimulus.copy()
        for n in range(len(cleaned_stimulus_a)):
            msk = np.isneginf(cleaned_stimulus_a[n])
            cleaned_stimulus_a[n, msk] = cleaned_stimulus_a[n, ~msk].min()
    
    elif silence_value is 'mean_row':
        cleaned_stimulus = []
        for row in whole_stimulus:
            row[np.isneginf(row)] = row[~np.isneginf(row)].mean()
            cleaned_stimulus.append(row)
        cleaned_stimulus_a = np.array(cleaned_stimulus)
    
    elif silence_value is 'median_row':
        cleaned_stimulus = []
        for row in whole_stimulus:
            row[np.isneginf(row)] = np.median(row[~np.isneginf(row)])
            cleaned_stimulus.append(row)
        cleaned_stimulus_a = np.array(cleaned_stimulus)
    
    elif silence_value is 'min_whole':
        cleaned_stimulus_a = whole_stimulus.copy()
        cleaned_stimulus_a[np.isneginf(cleaned_stimulus_a)] = \
            np.min(cleaned_stimulus_a[np.isfinite(cleaned_stimulus_a)])
        
    elif silence_value is 'mean_whole':
        cleaned_stimulus_a = whole_stimulus.copy()
        cleaned_stimulus_a[np.isneginf(cleaned_stimulus_a)] = \
            np.mean(cleaned_stimulus_a[np.isfinite(cleaned_stimulus_a)])

    elif silence_value is 'median_whole':
        cleaned_stimulus_a = whole_stimulus.copy()
        cleaned_stimulus_a[np.isneginf(cleaned_stimulus_a)] = \
            np.median(cleaned_stimulus_a[np.isfinite(cleaned_stimulus_a)])
    
    else:
        # blindly assign 'silence_value' to the munged values
        cleaned_stimulus_a = whole_stimulus.copy()
        cleaned_stimulus_a[np.isneginf(cleaned_stimulus_a)] = silence_value
    
    if z_score:
        for n in range(cleaned_stimulus_a.shape[0]):
            s = np.std(cleaned_stimulus_a[n, :])
            if s < 10**-6: print "warning, std too small"
            cleaned_stimulus_a[n, :] = (
                cleaned_stimulus_a[n, :] - cleaned_stimulus_a[n, :].mean()) / \
                np.std(cleaned_stimulus_a[n, :])
    
    return cleaned_stimulus_a


class RidgeFitter:
    """Like DirectFitter but operates on Experiment objects
    
    Gets the full matrices from it, cleans them if necessary
    Stores results
    """
    def __init__(self, expt=None):
        self.expt = expt
    
    def fit(self):
        #X = self.expt
        pass

class DirectFitter:
    """Calculates STRF for response matrix and stimulus matrix"""
    def __init__(self, X=None, Y=None):
        """New fitter
        X : (n_timepoints, n_features).
            I think this works better if mean of rows and cols is zero.
        Y : (n_timepoints, 1)
            Generally 0s and 1s.
        """
        self.X = X
        self.Y = Y
        self.XTX = None # pre-calculate this one
        
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("n_timepoints (dim0) is not the same!")
        if self.X.shape[1] > self.X.shape[1]:
            print "warning: more features than timepoints, possibly transposed"
    
    def STA(self):
        """Returns spike-triggered average of stimulus X and response Y.
        
        Therefore the STA is given by np.dot(X.transpose(), Y) / Y.sum().
        """
        return np.dot(self.X.transpose(), self.Y).astype(np.float) / \
            self.Y.sum()
        #X = X - X.mean(axis=0)[newaxis, :] # each feature has zero-mean over time
        #X = X - X.mean(axis=1)[:, newaxis] # each datapoint has zero-mean over features

    def whitened_STA(self, ridge_parameter=0.):
        if self.XTX is None:
            self.XTX = np.dot(self.X.transpose(), self.X)
        
        ridge_mat = ridge_parameter * len(self.XTX)**2 * np.eye(len(self.XTX))
        STA = self.STA()
    
        return np.dot(np.linalg.inv(self.XTX + ridge_mat), STA)*self.X.shape[0]


    