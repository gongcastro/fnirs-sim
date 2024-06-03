import numpy as np
from scipy.stats import gamma, norm
import matplotlib.pyplot as plt


def hrf(time: np.ndarray = np.arange(0, 20, 0.1),
        amplitude: float = 1.2,
        peak_delay: float = 5.00,
        peak_width: float = 0.50,
        ratio: float = 0.10,
        type: str = "hbo") -> np.ndarray:
    gamma = time ** peak_delay * np.exp(-time / peak_width)
    hrf = amplitude * gamma / gamma.max()
    signal = (ratio) * (hrf) if type == "hbo" else -ratio * 0.5 * hrf
    return signal


def sim_block(
        n_stim: int = 10,
        stim_duration: float = 0.8,
        isi: float = 1.5,
        silence_duration: int = 25):
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
    onsets = []
    offsets = []
    delay = 0
    for _ in np.arange(n_blocks):
        on, of = sim_block(**kwargs)
        onsets.append(on + delay)
        offsets.append(of + delay)
        delay = delay + on[1] + ibi
    return np.vstack(onsets), np.vstack(offsets)


def convolve_responses(times: np.ndarray,
                       values: np.ndarray,
                       sfreq: float = 10,
                       n: int = 25) -> np.ndarray:
    conv = []
    times = []
    for idx, (t, signal) in enumerate(zip(time_list, values)):
        pad1 = len(np.arange(0, np.min(t), 1/sfreq))
        pad2 = len(np.arange(np.max(t), np.max(time_list), 1/sfreq))
        s = np.hstack([np.zeros([pad1, ]), signal, np.zeros([pad2, ])])
        if idx != 0:
            s = s[0:np.min([len(x) for x in conv])]
        conv.append(s)
    conv_signal = np.sum(np.vstack(conv), 0)
    conv_times = np.linspace(0, np.max(time_list), len(conv_signal))
    return conv_signal, conv_times


def generate_noise(signal: np.ndarray,
                   time: np.ndarray,
                   type: str = "white",
                   sfreq: int = 10):
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
    hbo_list = []
    hbr_list = []
    time_list = []
    for o in onsets:
        t = np.arange(0, 20, 1/sfreq)
        pdist = norm(6, 0.2).rvs(1)
        adist = norm(0.50, 0.01).rvs(1)
        udist = norm(0.35, 0.1).rvs(1)
        hbo_list.append(hrf(t, pdist, adist, udist))
        hbr_list.append(hrf(t, pdist, adist, udist, type="hbr"))
        time_list.append(np.min(o) + t)

    hbo_conv, conv_times = convolve_responses(time_list, hbo_list)
    hbr_conv, _ = convolve_responses(time_list, hbr_list)
    heart_noise = generate_noise(hbo_conv,
                                 conv_times, type="heart")
    resp_noise = generate_noise(hbo_conv,
                                conv_times, type="respiration")
    mayer_noise = generate_noise(hbo_conv,
                                 conv_times, type="mayer")
    white_noise = generate_noise(hbo_conv, conv_times, type="white")
    hbo = np.sum([hbo_conv, white_noise, heart_noise,
                  resp_noise, mayer_noise], 0)
    hbr = np.sum([hbr_conv, white_noise, -heart_noise,
                  -resp_noise, -mayer_noise], 0)
    return hbo, hbr, conv_times


if __name__ == "__main__":

    sfreq = 10
    time = np.arange(0, 20, 1/sfreq)

    # simulate HRF and uncertainty
    iter = 100
    ddist = norm(6, 0.5).rvs(iter)
    wdist = norm(1, 0.1).rvs(iter)
    adist = norm(0.90, 0.1).rvs(iter)
    rdist = norm(0, 0.05).rvs(iter)
    hbo = np.vstack([hrf(time, a, d, w, type="hbo")
                     for (a, d, w) in zip(adist, ddist, wdist)])
    hbr = np.vstack([hrf(time, a, d, w, r, type="hbr")
                     for (a, d, w, r) in zip(adist, ddist, wdist, rdist)])
    plt.plot(time, np.transpose(hbo), c="b",
             alpha=10/iter, label="HbO")
    plt.plot(time, np.transpose(hbr), c="r",
             alpha=10/iter, label="HbR")
    plt.hlines(y=[0], xmin=0, xmax=20,
               color="grey", linestyles="dashed")
    plt.show()

    # paradigm params
    condition_labels = ["A", "B"]
    stim_duration = 270 * 3 / 1000
    words_per_block = 10
    n_blocks = 28
    isi = [1.5, 0.5]
    block_duration = np.sum(
        [stim_duration] * words_per_block +
        isi * int(words_per_block / 2))
    ibi = [25, 35]  # inter-block interval (silence)
    ibi_series = ibi * n_blocks
    run_duration = np.sum([block_duration] * n_blocks + ibi * n_blocks)

    onsets, offsets = sim_block()
    boxcar = np.hstack([[x, x, y, y] for (x, y) in zip(onsets, offsets)])
    plt.plot(boxcar, np.hstack([[0, 1, 1, 0]
                                for _ in np.arange(len(onsets))]))
    plt.show()

    # simulate events
    onset_design, offset_design = sim_design(n_blocks=n_blocks)
    block_onsets = np.min(onset_design, axis=1)
    block_offsets = np.min(offset_design, axis=1)
    y_onsets = np.vstack([np.repeat(-1, len(x)) for x in onset_design])
    y_offsets = np.vstack([np.repeat(1, len(x)) for x in offset_design])
    plt.scatter(onset_design, y_onsets, marker="|",
                label="Stimuli onsets", color="blue")
    plt.scatter(offset_design, y_offsets, marker="|",
                label="Stimuli offsets", color="orange")
    plt.vlines(block_onsets, ymin=-1.25, ymax=-0.75,
               label="Block onsets", colors="blue")
    plt.vlines(block_offsets, ymin=0.75, ymax=1.25,
               label="Block onsets", colors="orange")
    plt.legend(loc="upper right")
    plt.xlabel(xlabel="Time (s)")
    plt.ylim([-2, 2])
    plt.show()

    # add signal
    hbo_list = []
    hbr_list = []
    time_list = []
    time = np.arange(0, 20, 0.1)
    for o in onsets:
        adist = norm(0.50, 0.01).rvs(1)
        ddist = norm(6, 0.2).rvs(1)
        wdist = norm(0.35, 0.05).rvs(1)
        rdist = norm(-0.10, 0.05).rvs(1)
        hbo_list.append(hrf(time, adist, ddist, wdist, udist))
        hbr_list.append(hrf(time, adist, ddist, wdist, udist, type="hbr"))
        time_list.append(o + time)

        onsets, offsets = sim_block()
        boxcar = np.hstack([[x, x, y, y] for (x, y) in zip(onsets, offsets)])
        plt.plot(
            boxcar,
            np.hstack([[0, 1, 1, 0] for _ in np.arange(len(onsets))]),
            color="grey")

    [plt.plot(x, y, color="r", alpha=1/4)
     for (x, y) in zip(time_list, hbo_list)]
    [plt.plot(x, y, color="b", alpha=1/4)
     for (x, y) in zip(time_list, hbr_list)]

    hbo_conv, conv_times = convolve_responses(time_list, hbo_list)
    hbr_conv, _ = convolve_responses(time_list, hbr_list)

    plt.plot(conv_times, hbo_conv, color="red")
    plt.plot(conv_times, hbr_conv, color="blue")
    plt.show()

    # add measurement and heartbeat arctifacts
    heart_noise = generate_noise(hbo_conv,
                                 conv_times, type="heart")
    resp_noise = generate_noise(hbo_conv,
                                conv_times, type="respiration")
    mayer_noise = generate_noise(hbo_conv,
                                 conv_times, type="mayer")
    white_noise = generate_noise(hbo_conv, conv_times, type="white")
    hbo_noise = np.sum([hbo_conv, white_noise, heart_noise,
                       resp_noise, mayer_noise], 0)
    plt.plot(conv_times, white_noise, label="Measurement error")
    plt.plot(conv_times, heart_noise, label="Heartbeat")
    plt.plot(conv_times, resp_noise, label="Respiration", zorder=0)
    plt.plot(conv_times, mayer_noise, label="Mayer waves", zorder=0)
    plt.plot(conv_times, hbo_noise, label="Noisy signal")
    plt.plot(conv_times, hbo_conv, label="Clean HbO", zorder=0)
    plt.legend(loc="upper right")
    plt.xlabel(xlabel="Time (s)")
    plt.ylabel(ylabel="Haemoglobin (μM)")
    plt.show()

    hbo, hbr, times = sim_channel(onset_design)
    plt.plot(times, hbo, label="HbO", color="red")
    plt.plot(times, hbr, label="HbR", color="blue")
    plt.hlines(y=[0], xmin=0, xmax=np.max(times),
               color="grey", linestyles="dashed")
    plt.legend(loc="upper right")
    plt.xlabel(xlabel="Time (s)")
    plt.ylabel(ylabel="Haemoglobin (μM)")
    plt.show()
