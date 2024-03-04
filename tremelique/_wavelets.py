# Copyright (c) 2024 The Tremelique Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import abc
import copy

import numpy as np


class BaseWavelet(abc.ABC):

    def __init__(self, amp):
        self.amp = amp

    @abc.abstractmethod
    def __call__(self, time):
        pass

    def copy(self):
        return copy.deepcopy(self)


class GaussianWavelet(BaseWavelet):

    def __init__(self, amp, f_cut, delay=0):
        super().__init__(amp)
        self.f_cut = f_cut
        self.delay = delay

    def __call__(self, time):
        sqrt_pi = np.sqrt(np.pi)
        fc = self.f_cut / (3 * sqrt_pi)
        # Standard delay to make the wavelet start at time zero and be causal
        td = time - 2 * sqrt_pi / self.f_cut
        # Apply the user defined delay on top
        t = td - self.delay
        scale = self.amp / (2 * np.pi * (np.pi * fc) ** 2)
        res = scale * np.exp(-np.pi * (np.pi * fc * t) ** 2)
        return res


class RickerWavelet(BaseWavelet):

    def __init__(self, amp, f_cut, delay=0):
        super().__init__(amp)
        self.f_cut = f_cut
        self.delay = delay

    def __call__(self, time):
        sqrt_pi = np.sqrt(np.pi)
        fc = self.f_cut / (3 * sqrt_pi)
        # Standard delay to make the wavelet start at time zero and be causal
        td = time - 2 * sqrt_pi / self.f_cut
        # Apply the user defined delay on top
        t = td - self.delay
        scale = self.amp * (2 * np.pi * (np.pi * fc * t) ** 2 - 1)
        res = scale * np.exp(-np.pi * (np.pi * fc * t) ** 2)
        return res
