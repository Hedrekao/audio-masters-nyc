from ssspy.bss.ilrma import GaussILRMA as GaussILRMABase, TILRMA as TILRMABase, GGDILRMA as GGDILRMABase
from tqdm.notebook import tqdm

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