import numpy as np
from scipy.stats import gamma, norm
import matplotlib.pyplot as plt


def hrf(time: np.ndarray = np.arange(0, 20, 0.1),
        amplitude: float = 1.2,
        peak_delay: float = 5.00,
        peak_width: float = 0.50,
        ratio: float = 0.80) -> np.ndarray:
    """
    Hemodynamic Response Function (HRF)

    This function generates a single instance of an HRF. It is an adaptation of Glover et al. (1999)'s HRF function, which capitalises on the Gamma probabilistic distribution to generate a peak and an undershoot. Since we want to generate separate responses for HbO and HbR, only a peak is generated (no undershoot) for the HbO response. The HbR response is then generated as a transformation of the HbO response, by weighting its amplitude and changing its sign.

    Parameters
    ----------
    time : numpy.ndarray
        Array containing the time domain over which the HRF should be simulated, in seconds. Defaults to 0-20 seconds (which should be enough to observed a realistic HRF), with on oversampling of 0.1 (a time point will be generated very 0.1 seconds).
    peak_delay : float
        Time (in seconds) that takes the HRF to peak.
    peak_width : float
        Time (in seconds) that takes the HRF to go back to baseline.
    ratio : float
        This value will weight the amplitude of the HbO response to generate the HbR response. Values smaller than 1 will decrease the amplitude relative to the HbO response (this is usually more realistic), while values larger than 1 will increase its amplitude.

    Returns
    -------
    hbo : numpy.ndarray
        Simulated HRF for HbO
    hbr : numpy.ndarray
        Simulated HRF for HbR

    References
    ----------
    Glover, G. H. (1999). Deconvolution of impulse response in event-related BOLD fMRI1. Neuroimage, 9(4), 416-429. https://doi.org/10.1006/nimg.1998.0419

    Examples
    --------
    time = np.arange(0, 20, 0.1)
    hbo, hbr = hrf(time, ratio=0.25)
    """
    gamma = time ** peak_delay * np.exp(-time / peak_width)
    hbo = amplitude * gamma / gamma.max()
    hbr = -ratio * hbo
    return hbo, hbr


def sim_block(n_stim: int = 10,
              stim_duration: float = 0.8,
              isi: float = 1.5):
    """
    Simulate a block of stimuli.

    This function simulates a block of stimuli, returning two arrays. The first array contains the onsets of the stimuli in the block, the second contains their offsets. Onsets and offsets are generated taking into account the duration of each stimulus, the inter-stimulus interval, and the duration of the silences between stimuli. These parameters are set to simulate the block design by Gervain et al. (2012).

    Parameters
    ----------
    n_stim : int
        Number of stimuli in the block. Defaults to 10.
    stim_duration : float
        Duration of the stimulus in the block in seconds. Defaults to 0.8 seconds.
    isi : Float
        Inter-stimulus interval. Time elapsed between the offset of one stimulis and the onset of the next stimulus.

    Returns
    -------
    onsets : numpy.ndarray
        Stimulus onsets in seconds.
    offsets : numpy.ndarray
        Stimulus offsets.

    References
    ----------
    Gervain, J., Berent, I., & Werker, J. F. (2012). Binding at birth: The newborn brain detects identity relations and sequential position in speech. Journal of Cognitive Neuroscience, 24(3), 564-574. https://doi.org/10.1162/jocn_a_00157

    """
    block_onsets = []
    block_offsets = []
    for _ in np.arange(n_blocks):
        onsets = np.cumsum(
            np.repeat([stim_duration + isi], n_stim))
        onsets = np.insert(onsets[0:-1], 0, 0)
        offsets = onsets + isi
    return onsets, offsets


def sim_design(n_blocks: int = 28,
               ibi: int = 25,
               **kwargs):
    """
    Simulate the task design, with multiple blocks.

    This function concatenates several blocks, as generated by `sim_block` including silences between blocks, and returns two arrays: one containing the onset of each block, the other one containing the offset of each block. These parameters are set to simulate the block design by Gervain et al. (2012).

    Parameters
    ----------
    n_blocks : int
        Number of blocks. Defaults to 28.
    ibi : int
        Inter-block interval. Time elapsed between the offsets of one block and the onset of the next block. Defaults to 25.and
    **kwargs
        These parameters will be passed to the `sim_block` function.

    Returns
    -------
    onsets : numpy.ndarray
        Block onsets in seconds.
    offsets : numpy.ndarray
        Block offsets in seconds.

    References
    ----------
    Gervain, J., Berent, I., & Werker, J. F. (2012). Binding at birth: The newborn brain detects identity relations and sequential position in speech. Journal of Cognitive Neuroscience, 24(3), 564-574. https://doi.org/10.1162/jocn_a_00157
    """
    onsets = []
    offsets = []
    delay = 0
    for _ in np.arange(n_blocks):
        on, of = sim_block(**kwargs)
        onsets.append(on + delay)
        offsets.append(of + delay)
        delay = delay + on[1] + ibi
    return np.vstack(onsets), np.vstack(offsets)


def convolve_responses(times: list[np.ndarray],
                       values: list[np.ndarray],
                       sfreq: float = 10):
    """
    Convolve overlapping HRF responses.

    When simulating the HRF of a block desing, one HRF is generated for each stimulus. Depending on the timings, some HRF (if not all) may overlap to some degree. When analysing the HRF data of a block design, one usually wants to summarise the HRFs within the same block, either using the average or some sort of sum. To do this, the array containing the values of each HRF must be aggregated along with the corresponding values of the arrays of other HRF, i.e., which ocurr in the same time point. This is not necessarily trivial, since in real situations some arrays may be of different length.

    Since we have artificially generated the time course of the task, we can guess how many time points have elapsed before and ofter the beginning and end of a given response. This function pads zeroes at the beginning and end of each HRF array to make all arrays have the same length, while respectig the onset and offset of each response.

    Parameters
    ----------
    times : list of numpy.ndarray
        List containing the time course of each HRF response.
    values : list of numpy.ndarray
        List containing the HRF values of each response.
    sfreq : float
        Sampling frequency. Defaults to 10 Hz.

    Returns
    -------
    conv_signal : numpy.ndarray
        Convolved signal, resulting from summing all responses across the time domain.
    conv_time : numpy.ndarray
        Time domain of the convolved signal.
    """
    conv = []
    times = []
    for idx, (t, signal) in enumerate(zip(times, values)):
        pad1 = len(np.arange(0, np.min(t), 1/sfreq))
        pad2 = len(np.arange(np.max(t), np.max(times), 1/sfreq))
        s = np.hstack([np.zeros([pad1, ]),
                       signal,
                       np.zeros([pad2, ])])
        if idx != 0:
            s = s[0:np.min([len(x) for x in conv])]
        conv.append(s)
    conv_signal = np.sum(np.vstack(conv), 0)
    conv_time = np.linspace(0, np.max(times), len(conv_signal))
    return conv_signal, conv_time


def generate_noise(signal: np.ndarray,
                   time: np.ndarray,
                   type: str = "white",
                   sfreq: int = 10):
    """
    Add noise and physiological arctifacts to the HRF signal.

    This function adds up to four types of noise to the HRF signal:

    * White noise: measurement error, drawn from a normal distribution for each time point.
    * Heartbeat: adds a high-frequency sinewave that mimicks the heartbeat influence on the target HRF signal.
    * Respiration: adds a low-frequency sinewave that mimicks the respiration influence on the target HRF signal.
    * Mayer waves: adds a low-frequency sinewave that mimicks the Mayer waves influence on the target HRF signal.

    Parameters
    ----------
    signal : numpy.ndarray
        HRF signal
    time : numpy.ndarray
        Time domain of the signal in seconds.
    type : str {'white', 'heart' 'respiration', 'mayer'}
        Type of noise ("white") or physiological arctifact ("heartbeat", "respitation", "mayer").
    sfreq : float
        Samping frequency (defaults to 10 Hz).

    Returns
    -------
    numpy.ndarray
        Noisy signal.
    """
    n_samples = len(signal)
    if type == "white":
        return norm(0.0, 0.2).rvs(size=n_samples)
    else:
        if type == "heart":
            freq = np.random.normal(1.5, 0.2)
            inc = np.random.uniform(1.03, 1.10)
        if type == "respiration":
            freq = np.random.normal(0.25, 0.05)
            inc = np.random.uniform(1.03, 1.10)
        if type == "mayer":
            freq = np.random.normal(0.1, 0.02)
            inc = np.random.uniform(1.03, 1.10)
        return np.sin(time * freq / sfreq * np.pi * inc)


def sim_channel(onsets: np.ndarray, sfreq: float = 10):
    """
    Simulate a channels recorded HRF for a whole task.

    Parameters
    ----------
    onsets: numpy.ndarray
        Array containing stimulus onsets, as generated by `sim_design`.
    sfreq : float
        Samping frequency (defaults to 10 Hz).

    Returns
    -------
    hbo : numpy.ndarray
        HbO signal for the whole task.
    hrb : numpy.ndarray
        HbR signal for the whole task.
    times_conv : numpy.ndarray
        Time domain of the task.
    """
    hrf_list = []
    time_list = []
    for o in onsets:
        t = np.arange(0, 20, 1/sfreq)
        adist = norm(0.80, 0.15).rvs(1)
        ddist = norm(6, 0.5).rvs(1)
        wdist = norm(1, 0.15).rvs(1)
        rdist = norm(0.1, 0.15).rvs(1)
        hrf_list.append(hrf(t, adist, ddist, wdist, rdist))
        time_list.append(np.min(o) + t)

    hbo_list = []
    hbr_list = []
    for (hbo, hbr) in hrf_list:
        hbo_list.append(hbo)
        hbr_list.append(hbr)
    hbo_conv, times_conv = convolve_responses(time_list, hbo_list)
    hbr_conv, _ = convolve_responses(time_list, hbr_list)
    heart_noise = generate_noise(hbo_conv,
                                 times_conv, type="heart")
    resp_noise = generate_noise(hbo_conv,
                                times_conv, type="respiration")
    mayer_noise = generate_noise(hbo_conv,
                                 times_conv, type="mayer")
    white_noise = generate_noise(hbo_conv, times_conv, type="white")
    hbo = np.sum([hbo_conv, white_noise, heart_noise,
                  resp_noise, mayer_noise], 0)
    hbr = np.sum([hbr_conv, white_noise, -heart_noise,
                  -resp_noise, -mayer_noise], 0)
    return hbo, hbr, times_conv


if __name__ == "__main__":
    return None
