"""
Defines an engine to perform a self-consistent cycle of convergence studies over multiple input parameters.
One can normally use the simple `convergence` engine from `aiida-optimize` for single-parameter convergence studies, but this
engine allows to converge multiple parameters in a self-consistent way by reinjecting the
results of all previous convergence studies at each step.
Once the self-consistent cycle is converged, the engine outputs the final input parameters
that can be used for production calculations.
"""

import itertools
import typing as ty

from aiida_optimize import OptimizationWorkChain
from aiida_optimize.engines._result_mapping import Result
from aiida_optimize.engines.base import (
    OptimizationEngineImpl,
    OptimizationEngineWrapper,
)
from aiida_optimize.helpers import get_nested_result
from decorator import decorator
import numpy as np
from scipy.optimize import brentq

from aiida import orm
from aiida.engine import calcfunction
from aiida.orm.nodes.data.base import to_aiida_type

__all__ = ["SelfConsistentConvergence"]

ValueType = ty.Union[np.ndarray, float, int]


class _SelfConsistentConvergenceImpl(OptimizationEngineImpl):
    """
    Implementation class for the self-consistent convergence engine.
    """

    def __init__(
        self,
        input_keys: ty.List[str],
        input_values: ty.Dict[str, ty.List[ValueType]],
        result_key: str,
        conv_tols: ty.Dict[str, float],
        conv_windows: ty.Dict[str, int],
        scf_thrs: ty.Dict[str, float],
        scf_mixing: ty.Dict[str, float],
        scf_mixing_history: ty.Dict[str, int],
        scf_max_iterations: int,
        current_scf_iteration: int,
        current_conv_indices: ty.Dict[str, int],
        conv_results: ty.Dict[str, ty.List[ValueType]],
        scf_histories: ty.Dict[str, ty.List[ValueType]],
        finished_convergences: ty.Dict[str, bool],
        key_order: ty.List[str] = [],
        converged_inputs: ty.Dict[str, ValueType] = {},
        finished: bool = False,
        logger: ty.Optional[ty.Any] = None,
        result_state: ty.Optional[ty.Dict[int, Result]] = None,
    ):
        """
        Initialize the self-consistent convergence engine.

        :param input_keys: List of input parameter keys to be converged.
        :param input_values: Dictionary mapping input parameter keys to lists of values to test.
        :param result_key: Key to extract the result value from evaluation outputs.
        :param conv_tols: Dictionary mapping input parameter keys to convergence tolerances on the results.
        :param conv_windows: Dictionary mapping input parameter keys to convergence window sizes.
        :param scf_thrs: Dictionary mapping input parameter keys to SCF convergence thresholds on the input parameters.
        :param scf_mixing: Dictionary mapping input parameter keys to SCF mixing parameters.
        :param scf_mixing_history: Dictionary mapping input parameter keys to SCF mixing history sizes.
        :param scf_max_iterations: Maximum number of SCF iterations allowed.
        :param current_scf_iteration: Current SCF iteration count.
        :param current_conv_indices: Dictionary mapping input parameter keys to their current convergence study indices.
        :param conv_results: Dictionary mapping input parameter keys to lists of their convergence study results.
        :param scf_histories: Dictionary mapping input parameter keys to the history of their values during the SCF cycle.
        :param finished_convergences: Dictionary mapping input parameter keys to booleans indicating whether their convergence has finished (successfully or not).
        :param key_order: Optional list defining the order of submission of input parameters for evaluations, used to retrieve results in the correct order.
        :param converged_inputs: Dictionary mapping input parameter keys to their converged values if already found within the SCF cycle.
        :param finished: Boolean indicating whether the optimization is finished.
        :param logger: Optional logger instance for reporting.
        :param result_state: Optional serialized state of the result mapping.
        """
        super().__init__(logger=logger, result_state=result_state)
        self.input_keys = input_keys
        self.input_values = input_values
        self.result_key = result_key
        self.conv_tols = conv_tols
        self.conv_windows = conv_windows
        self.scf_thrs = scf_thrs
        self.scf_mixing = scf_mixing
        self.scf_mixing_history = scf_mixing_history
        self.scf_max_iterations = scf_max_iterations
        self.current_scf_iteration = current_scf_iteration
        self.current_conv_indices = current_conv_indices
        self.conv_results = conv_results
        self.scf_histories = scf_histories
        self.finished_convergences = finished_convergences
        self.key_order = key_order
        self.converged_inputs = converged_inputs
        self.finished = finished
        self._logger: OptimizationWorkChain

    # Override abstract properties and methods from base class
    @property
    def _state(self) -> ty.Dict[str, ty.Any]:
        """
        Generate the engine state, including variables
            input_keys, input_values, result_key, conv_tols, conv_windows,
            scf_thrs, scf_mixing, scf_mixing_history, scf_max_iterations,
            current_scf_iteration, current_conv_indices, conv_results,
            scf_histories, next_submit, next_update, finished
        and excluding variables
            logger
        """
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in ["_result_mapping", "_logger"]
        }

    @property
    def is_finished(self) -> bool:
        """
        Returns true if the optimization is finished.
        """
        return self.finished

    def _create_inputs(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> ty.List[ty.Dict[str, ty.Any]]:
        inputs_list = []
        self.key_order = []

        for key in self.input_keys:
            if self.finished_convergences[key]:
                continue
            elif self.is_parameter_converged(key):
                self.finished_convergences[key] = True
                self.converged_inputs[key] = self.input_values[key][
                    self.current_conv_indices[key]
                    - self.conv_windows[key]
                ]
                self._logger.report(
                    f"Parameter '{key}' converged at {self.converged_inputs[key]} in SCF iteration {self.current_scf_iteration} after {len(self.conv_results[key])} evaluations."
                )
                continue
            else:
                num_new_evals = self.num_new_evaluations(key)
                if num_new_evals == -1:
                    self.finished_convergences[key] = True
                    self._logger.report(
                        f"Parameter '{key}' could not converge within the provided input values in SCF iteration {self.current_scf_iteration}. Marking as finished."
                    )
                    continue
                start_idx = self.current_conv_indices[key]
                end_idx = start_idx + num_new_evals
                current_inputs = {
                    k: to_aiida_type(self.scf_histories[k][-1])
                    for k in self.input_keys
                }  # Use last SCF history values as base inputs
                for val in self.input_values[key][start_idx:end_idx]:
                    new_inputs = current_inputs.copy()
                    new_inputs[key] = to_aiida_type(
                        val  # Update only the current key with new value and keep others fixed
                    )
                    inputs_list.append(new_inputs)
                    self.key_order.append(key)
                self.current_conv_indices[key] += num_new_evals
                self._logger.report(
                    f"Parameter '{key}': launching {num_new_evals} new evaluations for values {', '.join(str(v) for v in self.input_values[key][start_idx:end_idx])}."
                )

        self._logger.report(
            f"Launching {len(inputs_list)} new evaluations for SCF iteration {self.current_scf_iteration}."
        )

        if (
            not self.scf_converged
            and self.all_parameters_converged
            and len(inputs_list) == 0
        ):
            self._logger.report(
                f"SCF cycle {self.current_scf_iteration} converged ..."
            )
            self.new_scf_iteration()
            self._logger.report(
                f"Starting SCF iteration {self.current_scf_iteration}:"
            )
            for key in self.input_keys:
                self._logger.report(
                    f"  Parameter '{key}':  \tcurrent value \t{self.scf_histories[key][-1]}"
                )

        self.check_finished()
        if self.finished and len(inputs_list) == 0:
            # If finished launch a single evaluation with the converged inputs to get final results
            self._logger.report(
                "Optimization finished, launching final evaluation with converged inputs:"
            )
            current_inputs = {
                k: to_aiida_type(self.scf_histories[k][-1])
                for k in self.input_keys
            }  # Use last SCF history values as base inputs
            for key in current_inputs:
                self._logger.report(
                    f"  Parameter '{key}':  \tvalue \t{current_inputs[key].value}"
                )
            return [current_inputs]

        return inputs_list

    def _update(
        self, outputs: ty.Dict[int, ty.Dict[str, orm.Node]]
    ) -> None:
        if self.finished:
            # If finished, just retrieve the final results
            pass
        else:
            results = [
                float(get_nested_result(output, self.result_key).value)
                for _, output in sorted(outputs.items())
            ]
            for param, result in zip(self.key_order, results):
                self.conv_results[param].append(result)

    def _get_optimal_result(self) -> ty.Tuple[int, ty.Any, ty.Any]:
        # get last result
        opt_index = max(self._result_mapping.keys())
        opt_inputs = self._result_mapping[opt_index].input

        keys = list(opt_inputs.keys())
        # Sort keys to have a deterministic order
        keys.sort()
        values = [opt_inputs[k] for k in keys]

        opt_input = orm.Dict(
            {
                "keys": orm.List(list=keys),
                "values": orm.List(list=values),
            }
        )
        opt_input.store()
        assert opt_input.is_stored
        opt_output = get_nested_result(
            self._result_mapping[opt_index].output, self.result_key
        )
        return (opt_index, opt_input, opt_output)

    # Additional methods specific to self-consistent convergence engine

    def get_distances(self, key: str) -> ty.List[float]:
        """
        Get the distances between the last values in the convergence window for a specific parameter.
        """
        window = self.conv_windows[key]
        values = self.conv_results[key][-window:]
        dist_triangle = map(
            lambda pair: float(
                np.linalg.norm(np.array(pair[0]) - np.array(pair[1]))
            ),
            itertools.combinations(values, 2),
        )
        return list(dist_triangle)

    def get_distance_triangle(
        self, key: str
    ) -> ty.List[ty.List[float]]:
        """
        Get the pair-wise distance triangle for the last values in the convergence window for a specific parameter.
        """
        window = self.conv_windows[key]
        values = self.conv_results[key][-window:]
        distance_triangle = [
            [
                float(
                    np.linalg.norm(
                        np.array(values[i]) - np.array(values[j])
                    )
                )
                for j in range(i + 1, len(values))
            ]
            for i in range(len(values) - 1)
        ]
        return distance_triangle

    def is_parameter_converged(self, key: str) -> bool:
        """
        Check if a specific parameter has converged based on the convergence criteria and distance triangle.
        """
        distances = self.get_distances(key)
        if not distances:
            return False
        tol = self.conv_tols[key]
        return np.max(distances) < tol

    @property
    def all_parameters_converged(self) -> bool:
        """
        Check if all parameters have converged.
        """
        return all(
            self.is_parameter_converged(key) for key in self.input_keys
        )

    @property
    def scf_converged(self) -> bool:
        """
        Check if the SCF cycle has converged based on the SCF thresholds.
        """
        for key in self.input_keys:
            history = self.scf_histories[key]
            if len(history) < 2:
                return False
            diff = float(
                np.linalg.norm(
                    np.array(history[-1]) - np.array(history[-2])
                )
            )
            if diff >= self.scf_thrs[key]:
                return False
        return True

    def generate_mixed_inputs(
        self, new_inputs: ty.Dict[str, ValueType]
    ) -> ty.Dict[str, ValueType]:
        """
        Generate mixed inputs for the SCF cycle using mixing parameters and history and a damping scheme.
        """
        mixed_inputs = {}
        for key in self.input_keys:
            history = self.scf_histories[key]
            mix_param = self.scf_mixing[key]
            mix_history_size = self.scf_mixing_history[key]
            relevant_history = history[-mix_history_size:]
            n = len(relevant_history)
            new_value = new_inputs[key]
            if n == 0:
                mixed_value = new_value
            elif n == 1:
                mixed_value = (
                    mix_param * new_value
                    + (1 - mix_param) * relevant_history[0]
                )
            else:
                f = lambda q: 1 - q * (1 - q**n) / (1 - q) - mix_param
                e = 1e-5
                q_opt = brentq(f, 0, 1 - e)
                assert isinstance(q_opt, float)
                mixed_value = mix_param * new_value
                for i, past_value in enumerate(
                    reversed(relevant_history)
                ):
                    mixed_value += past_value * (q_opt ** (i + 1))
            mixed_inputs[key] = mixed_value
        return mixed_inputs

    def new_scf_iteration(self) -> None:
        """
        Start a new SCF iteration by resetting the necessary variables.
        """
        self.current_scf_iteration += 1
        self.current_conv_indices = {key: 0 for key in self.input_keys}
        self.conv_results = {key: [] for key in self.input_keys}
        self.finished_convergences = {
            key: False for key in self.input_keys
        }
        mixed_inputs = self.generate_mixed_inputs(self.converged_inputs)
        self.converged_inputs = {}
        for key in self.input_keys:
            self.scf_histories[key].append(mixed_inputs[key])

    def num_new_evaluations(self, key: str) -> int:
        """
        Determine the number of new evaluations needed for a specific parameter to continue convergence.
        """
        if len(self.conv_results[key]) == 0:
            return self.conv_windows[key]
        distance_triangle = self.get_distance_triangle(key)
        num_new_evals = 0
        for i, row in enumerate(distance_triangle):
            if np.any(np.array(row) > self.conv_tols[key]):
                num_new_evals = i + 1
        if self.current_conv_indices[key] + num_new_evals > len(
            self.input_values[key]
        ):
            return -1
        return num_new_evals

    def check_finished(self) -> None:
        """
        Check if the optimization is finished based on convergence criteria.
        """
        if self.all_parameters_converged and self.scf_converged:
            self.finished = True
        elif self.current_scf_iteration >= self.scf_max_iterations:
            self.finished = True
        elif any(
            (
                self.finished_convergences[key]
                and (not key in self.converged_inputs)
            )
            for key in self.input_keys
        ):
            self.finished = True


class SelfConsistentConvergence(OptimizationEngineWrapper):
    """
    Wrapper class for the self-consistent convergence engine.
    """

    _IMPL_CLASS = _SelfConsistentConvergenceImpl

    def __new__(
        cls,
        input_keys: ty.List[str],
        input_values: ty.Dict[str, ty.List[ValueType]],
        result_key: str,
        conv_tols: ty.Dict[str, float],
        conv_windows: ty.Dict[str, int],
        scf_thrs: ty.Dict[str, float],
        scf_mixing: ty.Dict[str, float],
        scf_mixing_history: ty.Dict[str, int],
        scf_max_iterations: int,
        initial_inputs: ty.Dict[str, ValueType],
        logger: ty.Optional[ty.Any] = None,
    ) -> "SelfConsistentConvergence":
        """
        Create a new instance of the self-consistent convergence engine.

        :param input_keys: List of input parameter keys to be converged.
        :param input_values: Dictionary mapping input parameter keys to lists of values to test.
        :param result_key: Key to extract the result value from evaluation outputs.
        :param conv_tols: Dictionary mapping input parameter keys to convergence tolerances.
        :param conv_windows: Dictionary mapping input parameter keys to convergence window sizes.
        :param scf_thrs: Dictionary mapping input parameter keys to SCF convergence thresholds.
        :param scf_mixing: Dictionary mapping input parameter keys to SCF mixing parameters.
        :param scf_mixing_history: Dictionary mapping input parameter keys to SCF mixing history sizes.
        :param scf_max_iterations: Maximum number of SCF iterations allowed.
        :param initial_inputs: Dictionary mapping input parameter keys to their initial values as a first guess for the SCF cycle.
        :param logger: Optional logger instance for reporting.
        :return: An instance of SelfConsistentConvergence.
        """
        scf_histories = {
            key: [initial_inputs[key]] for key in input_keys
        }

        impl = cls._IMPL_CLASS(
            input_keys=input_keys,  # type: ignore
            input_values=input_values,  # type: ignore
            result_key=result_key,  # type: ignore
            conv_tols=conv_tols,  # type: ignore
            conv_windows=conv_windows,  # type: ignore
            scf_thrs=scf_thrs,  # type: ignore
            scf_mixing=scf_mixing,  # type: ignore
            scf_mixing_history=scf_mixing_history,  # type: ignore
            scf_max_iterations=scf_max_iterations,  # type: ignore
            finished_convergences={key: False for key in input_keys},  # type: ignore
            current_scf_iteration=0,  # type: ignore
            current_conv_indices={key: 0 for key in input_keys},  # type: ignore
            conv_results={key: [] for key in input_keys},  # type: ignore
            scf_histories=scf_histories,  # type: ignore
            logger=logger,
        )
        return impl  # type: ignore
