from ssspy.bss.ilrma import GaussILRMA as GaussILRMABase, TILRMA as TILRMABase, GGDILRMA as GGDILRMABase
from tqdm.notebook import tqdm
import numpy as np

class GaussILRMA(GaussILRMABase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.progress_bar = None

    def __call__(self, *args, n_iter: int = 100, **kwargs):
        self.n_iter = n_iter

        return super().__call__(*args, n_iter=n_iter, **kwargs)

    def update_once(self) -> None:
        if self.progress_bar is None:
            self.progress_bar = tqdm(total=self.n_iter)

        super().update_once()

        self.progress_bar.update(1)

class TILRMA(TILRMABase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.progress_bar = None

    def __call__(self, *args, n_iter: int = 100, **kwargs):
        self.n_iter = n_iter

        return super().__call__(*args, n_iter=n_iter, **kwargs)

    def update_once(self) -> None:
        if self.progress_bar is None:
            self.progress_bar = tqdm(total=self.n_iter)

        super().update_once()

        self.progress_bar.update(1)

class GGDILRMA(GGDILRMABase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.progress_bar = None

    def __call__(self, *args, n_iter: int = 100, **kwargs):
        self.n_iter = n_iter

        return super().__call__(*args, n_iter=n_iter, **kwargs)

    def update_once(self) -> None:
        if self.progress_bar is None:
            self.progress_bar = tqdm(total=self.n_iter)

        super().update_once()

        self.progress_bar.update(1)

def concatenate_arrays(*arrays):
    # Determine the maximum dimension along the concatenation axis
    max_dim = max(arr.shape[1] for arr in arrays)

    # Pad or reshape each array to match the maximum dimension
    padded_arrays = []
    for arr in arrays:
        if arr.shape[1] < max_dim:
            pad_width = ((0, 0), (0, max_dim - arr.shape[1]))
            arr_padded = np.pad(arr, pad_width, mode='constant', constant_values=0)
            padded_arrays.append(arr_padded)
        else:
            padded_arrays.append(arr)

    # Concatenate the modified arrays
    result = np.concatenate(padded_arrays, axis=0)

    return result