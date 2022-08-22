#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#


class AbstractAnsatz:

    __slots__ = []

    def _weight(self, configuration):
        """
        Calculate the weight of the given configuration.

        Parameters
        ----------
        configuration : Configuration
            The given configuration to calculate weight.

        Returns
        -------
        complex | float
            The result weight.
        """
        raise NotImplementedError("weight not implemented")

    def _delta(self, configuration):
        """
        Calculate the delta of the given configuration.

        Parameters
        ----------
        configuration : Configuration
            The given configuration to calculate delta.

        Returns
        -------
        Delta
            The delta object, which will be called by allreduce and apply_gradient.
        """
        raise NotImplementedError("delta not implemented")

    def weight_and_delta(self, configurations, calculate_delta):
        """
        Calculate the weight and delta of the given configurations.

        Parameters
        ----------
        configurations : list[configuration]
            The given configuration list to calculate weight and delta.
        calculate_delta : bool
            Whether to calculate delta.

        Returns
        -------
        tuple[list[float | complex], None | list[Delta]]
        """
        weight = [self._weight(configuration) for configuration in configurations]
        if calculate_delta:
            delta = [self._delta(configuration) for configuration in configurations]
        else:
            delta = None
        return weight, delta

    def refresh_auxiliaries(self):
        """
        Refresh auxiliaries after updating state.
        """
        raise NotImplementedError("refresh auxiliaries not implemented")

    def ansatz_prod_sum(self, a, b):
        """
        Calculate the summary of product of two ansatz like data. If None is given, the data ansatz itself stored will
        be used.

        Parameters
        ----------
        a, b : Delta
            The two ansatz like data.

        Returns
        -------
        float
            The dot of a and b.
        """
        raise NotImplementedError("ansatz_prod_sum not implemented")

    def ansatz_conjugate(self, a):
        """
        Calculate the conjugate of a ansatz like data, If None is given, the data ansatz itself stored will be used.

        Parameters
        ----------
        a : Delta
            The ansatz like data.

        Returns
        -------
        Delta
            The conjugate
        """
        raise NotImplementedError("ansatz_conjugate not implemented")

    def ansatz_dot(self, a, b):
        """
        Calculate the dot of two ansatz like data. If None is given, the data ansatz itself stored will be used.

        Parameters
        ----------
        a, b : Delta
            The two ansatz like data.

        Returns
        -------
        float
            The dot of a and b.
        """
        return self.ansatz_prod_sum(self.ansatz_conjugate(a), b).real

    def buffers(self, delta):
        """
        Get buffers of this ansatz.

        Yields
        ------
        buffer
            The buffers of this ansatz.
        """
        raise NotImplementedError("buffers not implemented")

    def elements(self, delta):
        """
        Get elements of this ansatz.

        Yields
        ------
        float | complex
            The elements of this ansatz.
        """
        raise NotImplementedError("elements not implemented")

    def buffer_count(self, delta):
        """
        Get buffer count of this ansatz.

        Returns
        -------
        int
            The buffer count of this ansatz.
        """
        raise NotImplementedError("buffer count not implemented")

    def element_count(self, delta):
        """
        Get element count of this ansatz.

        Returns
        -------
        int
            The element count of this ansatz.
        """
        raise NotImplementedError("element count not implemented")

    def buffers_for_mpi(self, delta):
        """
        Get buffers of this ansatz, which can be used in mpi.

        Yields
        ------
        buffer
            The buffers of this ansatz.
        """
        raise NotImplementedError("buffers for mpi not implemented")

    def recovery_real(self, delta=None):
        """
        Recovery the input to real. If None is given, return True if and only if it is needed to reocovery to real data.
        """
        if delta is None:
            return False
        raise RuntimeError("Program should never reach here")

    def normalize_ansatz(self):
        """
        Normalize this ansatz.
        """
        pass
