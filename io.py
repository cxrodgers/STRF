"""STRFlab-type format spec

Each stimulus is labeled with a string suffix, such as '0001'.
Each stimulus has 3 corresponding files:
    Stimulus files, eg stim0001.wav
        * A wav file
    Spike files, eg spike0001
        * Plain-text
        * Each line of the file is the spike times, in seconds, from a single
          trial. Each spike time is separated with a space.
        * Each trial can start at time 0.0, or not. The presence of an
          interval file indicates which of these is the case.
        * The number of newline characters is exactly equal to the number of
          trials.
    Interval files, eg interval0001
        * Plain text
        * Each line of the file is the start time of the trial, then a space,
          then the stop time of the trial, in seconds.
        * Each trial can start at time 0.0, or not.
        * The number of newline characters is exactly equal to the number of
          trials.
"""

import os, re
import numpy as np
import kkpandas # could consider making this optional

class STRFlabFileSchema:
    """Object encapsulating format spec for STRFlab-type files"""
    stim_file_prefix = 'stim'
    interval_file_prefix = 'interval'
    spike_file_prefix = 'spike'
    stim_file_regex = r'stim(\d+)\.wav'
    spike_file_regex = r'^spike(\d+)$'
    interval_file_regex = r'^interval(\d+)$'
    
    def __init__(self, directory):
        self.directory = os.path.abspath(directory)
        self.name = os.path.split(self.directory)[1]
        self.populate()
    
    def populate(self):
        all_files = os.listdir(self.directory)
        
        # Find files matching regexes
        self.spike_file_labels = apply_and_filter_by_regex(
            self.spike_file_regex, all_files, sort=True)
        self.interval_file_labels = apply_and_filter_by_regex(
            self.interval_file_regex, all_files, sort=True)
        
        # Reconstruct the filenames that match
        self.spike_filenames = [self.spike_file_prefix + label 
            for label in self.spike_file_labels]
        self.interval_filenames = [self.interval_file_prefix + label 
            for label in self.interval_file_labels]            

        self._force_reload = False

def apply_and_filter_by_regex(pattern, list_of_strings, sort=True):
    """Apply regex pattern to each string, return first hit from each match"""
    res = []
    for s in list_of_strings:
        m = re.match(pattern, s)
        if m is None:
            continue
        else:
            res.append(m.groups()[0])
    if sort:
        return sorted(res)
    else:
        return res

def parse_space_sep(s, dtype=np.int):
    """Returns a list of integers from a space-separated string"""
    s2 = s.strip()
    if s2 == '':
        return []    
    else:
        return [dtype(ss) for ss in s2.split()]

def read_directory(directory, **folding_kwargs):
    """Return dict {suffix : folded} for all stimuli in the directory"""
    sls = STRFlabFileSchema(directory)
    assert np.all(
        np.asarray(sls.spike_file_labels) == 
        np.asarray(sls.interval_file_labels))
    
    dfolded = {}
    for suffix in sls.spike_file_labels:
        dfolded[suffix] = read_single_stimulus(directory, suffix, **folding_kwargs)
    
    return dfolded
    

def read_single_stimulus(directory, suffix, **folding_kwargs):
    """Read STRFlab-type text files for a single stimulus
    
    TODO:
    split this into smaller functions, to read just spikes or just intervals
    eventually these should become properties in some Reader object
    """
    # Load the spikes
    spikes_filename = os.path.join(directory, 'spike' + suffix)
    with file(spikes_filename) as fi:
        lines = fi.readlines()
    spike_times = [parse_space_sep(s, dtype=np.float)   
        for s in lines]

    # Load the intervals
    intervals_filenames = os.path.join(directory, 'interval' + suffix)
    with file(intervals_filenames) as fi:
        lines = fi.readlines()
    intervals = np.asarray([parse_space_sep(s, dtype=np.float) 
        for s in lines])
    starts, stops = intervals.T

    # Create the folded
    folded = kkpandas.Folded(spike_times, starts=starts, stops=stops,
        **folding_kwargs)
    return folded


def write_single_stimulus(suffix, spikes, starts=None, stops=None,
    output_directory='.', time_formatter=None, write_intervals=True,
    flush_directory=True):
    """Write out STRFlab-type text files for a single stimulus.
    
    suffix : string labelling this stimulus
        This will be appended to the filenames.
        ie '0001' for spike0001
    
    spikes : list of arrays of spike times, or kkpandas.Folded
        These can be starting from zero for each trial, or not.
    
    starts : array-like, start times of each trial
        If None, use spikes.starts
        TODO:
        If None, use 0.0 for all trials. In this case you should probably
        reference your spike times to the beginning of their trial.
    
    stops : array-like, stop times of each trial
        If None, use spikes.stops.
        TODO:
        If None, use 0.0 for all trials. Probably not too useful.
    
    output_directory : where to write the files
    
    time_formatter : function to apply to each time to turn it into a string    
        The default is a floating point formatter with 5 digits after the
        decimal point.
    
    write_intervals : if False, just write the spike files

    flush_directory : erase everything in the directory before writing
    
    The following error checking is done:
    1)  The lengths of `spikes`, `starts`, and `stops` should be the same
    2)  The spike times on each trial must fall in between the start time
        and stop time of that trial.
        TODO:
        currently this is a strict < and > test. One of these should allow
        equality, probably the start.
    """
    # Set defaults
    if starts is None and write_intervals:
        starts = spikes.starts
    if stops is None and write_intervals:
        stops = spikes.stops
    if time_formatter is None:
        time_formatter = lambda v: '%.5f' % v
    
    # Set filenames
    spike_filename = os.path.join(output_directory, 'spike' + suffix)
    interval_filename = os.path.join(output_directory, 'interval' + suffix)
    
    # error check lengths
    if write_intervals:
        assert len(spikes) == len(starts), \
            "Length of spikes must equal length of starts"
        assert len(spikes) == len(stops), \
            "Length of spikes must equal length of stops"
        
        # error check ordering
        for trial_start, trial_stop, trial_spikes in zip(
            starts, stops, spikes):
            
            assert np.all(np.asarray(trial_spikes) > trial_start), \
                "Some spikes fall before trial start"
            assert np.all(np.asarray(trial_spikes) < trial_stop), \
                "Some spikes fall after trial stop"

    # Set up output directory
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    
    # Write the spikes for each repetition
    to_write = []
    for trial_spikes in spikes:
        to_write.append(' '.join(map(time_formatter, trial_spikes)))
    with file(spike_filename, 'w') as fi:
        fi.write("\n".join(to_write))

    
    # Write the start and stop time of each repetition
    if write_intervals:
        to_write = []
        for trial_start, trial_stop in zip(starts, stops):
            to_write.append(time_formatter(trial_start) + ' ' + 
                time_formatter(trial_stop))
        
        with file(interval_filename, 'w') as fi:
            fi.write("\n".join(to_write))
