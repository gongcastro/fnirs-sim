import numpy as np
import matplotlib.pyplot as plt


class Stim:
    """
    A stimulus or trigger.
    """

    def __init__(self, onset: float = 0, duration: float = 0.80):
        self.onset: float = onset
        self.offset: float = onset + duration

    def __repr__(self):
        return f"Stim({self.onset:.2f}, {self.offset:.2f})"


class Block:
    """
    A block of stimuli.
    """

    def __init__(self,
                 stim: list[Stim],
                 onset: float = 0.,
                 isi: float or list[float] = 1.5,
                 condition: str or int = "A"):
        stim_list, on_list, of_list = [], [], []
        for idx, s in enumerate(stim):
            duration = s.offset - s.onset
            if idx == 0:
                on = s.onset + onset
                of = s.offset + onset
                stim_list.append(Stim(on, duration))
                on_list.append(on)
                of_list.append(of)
            else:
                on = of_list[idx-1] + isi
                of = on + duration
                stim_list.append(Stim(on, duration))
                on_list.append(on)
                of_list.append(of)
        self.onset = np.min(on_list)
        self.offset = np.max(of_list)
        self.stim = stim_list
        self.condition = condition

    def __repr__(self):
        return f"{self.__class__.__name__}({self.onset:.2f}, {
            self.offset:.2f}, {self.stim})"

    def plot(self, **kwargs):
        onsets = [s.onset for s in self.stim]
        offsets = [s.offset for s in self.stim]
        x = np.hstack([[x, x, y, y] for (x, y) in zip(onsets, offsets)])
        y = np.hstack([[0, 1, 1, 0] for _ in np.arange(len(onsets))])
        plt.plot(x, y, **kwargs)
        plt.show()


class Design:
    """
    A block design.
    """

    def __init__(self,
                 blocks: list[Block],
                 ibi: float or list[float] = 25.):
        block_list, on_list, of_list = [], [], []
        for idx, b in enumerate(blocks):
            duration = b.offset - b.onset
            if idx == 0:
                on = b.onset
                of = b.offset
                block_list.append(b)
                on_list.append(on)
                of_list.append(of)
            else:
                on = of_list[idx-1] + ibi
                of = on + duration
                block_list.append(
                    Block(b.stim,
                          onset=on,
                          condition=b.condition))
                on_list.append(on)
                of_list.append(of)
        self.onset = np.min(on_list)
        self.offset = np.max(of_list)
        self.blocks = block_list

    def __repr__(self):
        return f"{self.__class__.__name__}({self.onset:.2f}, {
            self.offset:.2f}, {self.blocks})"

    def plot(self, **kwargs):
        conditions = np.unique([b.condition for b in self.blocks])
        for c in conditions:
            onsets = [b.onset for b in self.blocks if b.condition == c]
            offsets = [b.offset for b in self.blocks if b.condition == c]
            x = np.hstack([[x, x, y, y] for (x, y) in zip(onsets, offsets)])
            y = np.hstack([[0, 1, 1, 0] for _ in np.arange(len(onsets))])
            plt.plot(x, y, **kwargs)
        plt.show()
