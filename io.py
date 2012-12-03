import os.path
import numpy as np


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

def write_single_stimulus(suffix, spikes, starts=None, stops=None,
    output_directory='.', time_formatter=None, write_intervals=True):
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
    
    # Write the spikes for each repetition
    with file(spike_filename, 'w') as fi:
        for trial_spikes in spikes:
            fi.write(' '.join(map(time_formatter, trial_spikes)))
            fi.write('\n')
    
    # Write the start and stop time of each repetition
    if write_intervals:
        with file(interval_filename, 'w') as fi:
            for trial_start, trial_stop in zip(starts, stops):
                fi.write(
                    time_formatter(trial_start) + ' ' + 
                    time_formatter(trial_stop) + '\n')