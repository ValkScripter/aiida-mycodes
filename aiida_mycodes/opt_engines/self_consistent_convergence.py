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

from aiida_optimize.engines._result_mapping import Result
from aiida_optimize.engines.base import (
    OptimizationEngineImpl,
    OptimizationEngineWrapper,
)
from aiida_optimize.helpers import get_nested_result
import numpy as np

from aiida import orm
from aiida.orm.nodes.data.base import to_aiida_type

__all__ = ["SelfConsistentConvergence"]


class _SelfConsistentConvergenceImpl(OptimizationEngineImpl):
    """
    Implementation class for the self-consistent convergence engine.

    This engine performs convergence studies on multiple input parameters in a self-consistent manner.
    At each SCF iteration, it cycles through all parameters, converging each one while keeping the
    others fixed at their current best values. The process continues until all parameters are converged
    within the specified SCF threshold.
    """

    def __init__(
        self,
        *,
        input_values: ty.Dict[str, ty.List[ty.Any]],
        tol: ty.Dict[str, float],
        scf_thr: ty.Dict[str, float],
        result_key: str,
        convergence_window: ty.Dict[str, int],
        array_name: ty.Optional[str],
        max_scf_iterations: int,
        current_scf_iteration: int,
        param_order: ty.List[str],
        converged_values: ty.Dict[str, ty.Any],
        current_indices: ty.Dict[str, int],
        result_values: ty.Dict[str, ty.List[ty.Any]],
        scf_history: ty.List[ty.Dict[str, ty.Any]],
        params_converged: ty.Dict[str, bool],
        initialized: bool,
        logger: ty.Optional[ty.Any],
        result_state: ty.Optional[ty.Dict[int, Result]] = None,
    ):
        super().__init__(logger=logger, result_state=result_state)
        self.input_values = input_values
        self.tol = tol
        self.scf_thr = scf_thr
        self.result_key = result_key
        self.array_name = array_name
        self.convergence_window = convergence_window
        self.max_scf_iterations = max_scf_iterations
        self.current_scf_iteration = current_scf_iteration
        self.param_order = param_order
        self.converged_values = converged_values
        self.current_indices = current_indices
        self.result_values = result_values
        self.scf_history = scf_history
        self.params_converged = params_converged
        self.initialized = initialized

    @property
    def _state(self) -> ty.Dict[str, ty.Any]:
        """
        Generate the engine state, excluding logger and result_mapping.
        """
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in ["_result_mapping", "_logger"]
        }

    def _all_params_converged(self) -> bool:
        """Check if all parameters in the current SCF cycle are converged."""
        return all(self.params_converged.values())

    def _convert_result_to_native(self, result: ty.Any) -> ty.Any:
        """Convert AiiDA result to python/numpy object."""
        if isinstance(result, orm.ArrayData):
            return result.get_array(self.array_name)
        elif isinstance(result, (orm.Float, orm.Int)):
            return result.value
        return result

    def _get_result_window(self, param: str) -> ty.List[ty.Any]:
        """
        Create a list of results corresponding to the current convergence window
        for a specific parameter.
        """
        window_size = self.convergence_window[param]
        result_window = self.result_values[param][-window_size:]
        return [
            self._convert_result_to_native(r) for r in result_window
        ]

    def _calculate_distance_triangle(
        self, param: str
    ) -> ty.List[ty.List[ty.Any]]:
        """
        Calculate the pair-wise distance between entries in the convergence window
        for a specific parameter.
        """
        result_window = self._get_result_window(param)
        distance_triangle = [
            [
                np.linalg.norm(result_window[i] - result_window[j])
                for j in range(i + 1, len(result_window))
            ]
            for i in range(len(result_window) - 1)
        ]
        return distance_triangle

    def _num_new_iters(self, param: str) -> int:
        """
        Determine the minimum number of additional outputs needed to potentially
        converge the given parameter.
        """
        window_size = self.convergence_window[param]
        if len(self.result_values[param]) < window_size:
            return window_size - len(self.result_values[param])

        distance_triangle = self._calculate_distance_triangle(param)
        num_new_iters = 0
        for i, row in enumerate(distance_triangle):
            if np.any(np.array(row) > self.tol[param]):
                num_new_iters = i + 1

        # Check if we can get enough new iterations
        if self.current_indices[param] + num_new_iters > len(
            self.input_values[param]
        ):
            return -1

        return num_new_iters

    def _is_param_converged(self, param: str) -> bool:
        """
        Check if a specific parameter has converged by calculating distances
        within the convergence window.
        """
        window_size = self.convergence_window[param]
        if len(self.result_values[param]) < window_size:
            return False

        distance_triangle = self._calculate_distance_triangle(param)
        distances = list(itertools.chain(*distance_triangle))

        return bool(np.max(distances) < self.tol[param])

    def _is_scf_converged(self) -> bool:
        """
        Check if self-consistency is reached by comparing converged values
        between consecutive SCF iterations.
        """
        if self.current_scf_iteration < 1 or len(self.scf_history) < 2:
            return False

        prev_values = self.scf_history[-2]
        curr_values = self.scf_history[-1]

        for param in self.param_order:
            prev_val = self._convert_result_to_native(
                prev_values[param]
            )
            curr_val = self._convert_result_to_native(
                curr_values[param]
            )

            distance = np.linalg.norm(prev_val - curr_val)
            if distance >= self.scf_thr[param]:
                return False

        return True

    @property
    def is_converged(self) -> bool:
        """
        Check if the self-consistent convergence cycle is complete.
        """
        if not self.initialized:
            return False

        return self._is_scf_converged()

    @property
    def is_finished(self) -> bool:
        """
        Check if the optimization is finished, either by convergence or
        by reaching maximum iterations.
        """
        if not self.initialized:
            return False

        # Reached maximum SCF iterations
        if self.current_scf_iteration >= self.max_scf_iterations:
            self._logger.logger.warning(
                f"Maximum SCF iterations ({self.max_scf_iterations}) reached without convergence."
            )
            return True

        # Check if we've exhausted input values for any parameter
        for param in self.param_order:
            if len(self.result_values[param]) >= len(
                self.input_values[param]
            ):
                if not self._is_param_converged(param):
                    self._logger.logger.warning(
                        f"Exhausted all input values for parameter '{param}' without convergence."
                    )
                    return True

        return self.is_converged

    @property
    def is_finished_ok(self) -> bool:
        """Check if finished successfully with convergence."""
        return self.is_finished and self.is_converged

    def _create_inputs(self) -> ty.List[ty.Dict[str, orm.Node]]:
        """
        Create the inputs for the evaluation function.
        Converges all parameters in parallel, each varying independently
        while others are fixed at their current best values.
        """
        if not self.initialized:
            # Initialize: fill convergence window for all parameters
            self.initialized = True
            # Use maximum window size to ensure all parameters can be checked
            num_new_iters = max(self.convergence_window.values())
            self._logger.logger.info(
                f"Initializing self-consistent convergence for parameters: {', '.join(self.param_order)}"
            )
            self._logger.logger.info(
                f"Convergence windows: {self.convergence_window}"
            )
            self._logger.logger.info(f"Tolerances: {self.tol}")
            self._logger.logger.info(f"SCF thresholds: {self.scf_thr}")
        else:
            # Check if all parameters converged - move to next SCF iteration
            if self._all_params_converged():
                # Store SCF history
                scf_snapshot = {}
                for param in self.param_order:
                    if self.result_values[param]:
                        window_size = self.convergence_window[param]
                        opt_index = (
                            len(self.result_values[param])
                            - window_size // 2
                            - 1
                        )
                        scf_snapshot[param] = self.result_values[param][
                            opt_index
                        ]
                self.scf_history.append(scf_snapshot)

                self._logger.logger.info(
                    f"SCF iteration {self.current_scf_iteration} completed. All parameters converged."
                )

                # Log converged values for this iteration
                for param in self.param_order:
                    val = self.converged_values[param]
                    self._logger.logger.info(f"  {param}: {val}")

                # Check SCF convergence
                if self._is_scf_converged():
                    self._logger.logger.info(
                        "Self-consistent convergence achieved! SCF cycle converged."
                    )
                    return []  # Converged, no more inputs needed

                # Start new SCF iteration
                self.current_scf_iteration += 1
                self._logger.logger.info(
                    f"Starting SCF iteration {self.current_scf_iteration}"
                )

                # Log SCF distances if history available
                if len(self.scf_history) >= 2:
                    prev_values = self.scf_history[-2]
                    curr_values = self.scf_history[-1]
                    self._logger.logger.info("  SCF distances:")
                    for param in self.param_order:
                        prev_val = self._convert_result_to_native(
                            prev_values[param]
                        )
                        curr_val = self._convert_result_to_native(
                            curr_values[param]
                        )
                        distance = np.linalg.norm(prev_val - curr_val)
                        self._logger.logger.info(
                            f"    {param}: {distance:.6e} (threshold: {self.scf_thr[param]:.6e})"
                        )

                for param in self.param_order:
                    self.params_converged[param] = False
                    self.result_values[param] = []

                num_new_iters = max(self.convergence_window.values())
            else:
                # Continue convergence for unconverged parameters
                num_new_iters = 1

                # Log progress of unconverged parameters
                unconverged = [
                    p for p, c in self.params_converged.items() if not c
                ]
                if unconverged:
                    self._logger.logger.info(
                        f"SCF iteration {self.current_scf_iteration}: Continuing convergence for {', '.join(unconverged)}"
                    )

        # Create inputs for all unconverged parameters
        inputs = []

        for param in self.param_order:
            if self.params_converged[param]:
                continue  # Skip already converged parameters

            # Determine how many iterations needed for this parameter
            param_iters = self._num_new_iters(param)

            if param_iters == -1:
                # Cannot converge with remaining inputs
                self._logger.logger.warning(
                    f"Cannot converge parameter '{param}' with available input values. "
                    f"Used {len(self.result_values[param])}/{len(self.input_values[param])} values."
                )
                self.params_converged[param] = (
                    True  # Mark as converged to skip
                )
                continue

            if param_iters == 0:
                # Already has enough results
                continue

            # Use the smaller of param_iters or num_new_iters
            iters_to_add = min(param_iters, num_new_iters)

            start_idx = self.current_indices[param]
            end_idx = start_idx + iters_to_add

            self._logger.logger.debug(
                f"Adding {iters_to_add} iteration(s) for parameter '{param}' "
                f"(indices {start_idx} to {end_idx-1})"
            )

            for i in range(start_idx, end_idx):
                input_dict = {}

                # Add the varying parameter
                input_dict[param] = to_aiida_type(
                    self.input_values[param][i]
                )

                # Add all other parameters at their converged values
                for other_param in self.param_order:
                    if (
                        other_param != param
                        and other_param in self.converged_values
                    ):
                        input_dict[other_param] = to_aiida_type(
                            self.converged_values[other_param]
                        )

                inputs.append(input_dict)

            self.current_indices[param] = end_idx

        return inputs

    def _update(self, outputs: ty.Dict[int, ty.Any]) -> None:
        """
        Update the state of the engine by saving output values from
        parallel convergence studies of all parameters.
        """
        # Group outputs by parameter based on which parameter was varied
        output_keys = sorted(outputs.keys())

        for key in output_keys:
            output = outputs[key]
            result = get_nested_result(output, self.result_key)

            # Determine which parameter this output corresponds to
            # by checking which input parameter was used
            input_data = self._result_mapping[key].input

            # Find which parameter was being varied for this calculation
            # by identifying which parameter has a value NOT in converged_values
            # or has a different value than the converged one
            varied_param = None

            for param in self.param_order:
                if param not in input_data:
                    continue

                input_val = input_data[param]
                if isinstance(input_val, (orm.Float, orm.Int)):
                    input_val = input_val.value

                # Check if this parameter was being varied
                if param not in self.converged_values:
                    # No converged value yet, this could be the varied parameter
                    varied_param = param
                    break
                else:
                    # Compare with converged value
                    converged_val = self.converged_values[param]
                    if not np.allclose(input_val, converged_val):
                        varied_param = param
                        break

            # If all parameters match converged values, it's from the first SCF iteration
            # In that case, determine by checking against input_values sequence
            if varied_param is None:
                for param in self.param_order:
                    if (
                        param in input_data
                        and not self.params_converged[param]
                    ):
                        input_val = input_data[param]
                        if isinstance(input_val, (orm.Float, orm.Int)):
                            input_val = input_val.value

                        # Check if this matches the next expected value
                        expected_idx = len(self.result_values[param])
                        if expected_idx < len(self.input_values[param]):
                            expected_val = self.input_values[param][
                                expected_idx
                            ]
                            if np.allclose(input_val, expected_val):
                                varied_param = param
                                break

            # Store the result for the identified parameter
            if varied_param is not None:
                self.result_values[varied_param].append(result)

        # Check convergence status for each parameter
        for param in self.param_order:
            if not self.params_converged[
                param
            ] and self._is_param_converged(param):
                # Store converged value (use middle of convergence window)
                window_size = self.convergence_window[param]
                opt_index = (
                    len(self.result_values[param])
                    - window_size // 2
                    - 1
                )
                self.converged_values[param] = (
                    self._convert_result_to_native(
                        self.result_values[param][opt_index]
                    )
                )
                self.params_converged[param] = True

                # Calculate max distance in convergence window for logging
                distance_triangle = self._calculate_distance_triangle(
                    param
                )
                distances = list(itertools.chain(*distance_triangle))
                max_distance = np.max(distances) if distances else 0.0

                self._logger.logger.info(
                    f"Parameter '{param}' converged to {self.converged_values[param]} "
                    f"(max distance: {max_distance:.6e}, tolerance: {self.tol[param]:.6e})"
                )

    def _get_optimal_result(
        self,
    ) -> ty.Tuple[int, ty.Dict[str, orm.Node], ty.Dict[str, orm.Node]]:
        """
        Retrieve the converged input and output values for all parameters.

        Returns
        -------
        Tuple[int, Dict[str, orm.Node], Dict[str, orm.Node]]
            - Index of the last result in the result mapping (always returns last index)
            - Dictionary mapping parameter names to their converged input values (as AiiDA nodes)
            - Dictionary mapping parameter names to their converged output values (as AiiDA nodes)
        """
        optimal_inputs = {}
        optimal_outputs = {}

        # Get the last SCF iteration results
        if self.scf_history:
            last_scf = self.scf_history[-1]
            for param in self.param_order:
                optimal_inputs[param] = to_aiida_type(
                    self.converged_values[param]
                )
                optimal_outputs[param] = last_scf[param]

        # Return the index of the last evaluation in the result mapping
        # For multi-parameter optimization, we return the last index as the "optimal" one
        opt_index = (
            len(self._result_mapping) - 1
            if len(self._result_mapping) > 0
            else 0
        )

        return (opt_index, optimal_inputs, optimal_outputs)


class SelfConsistentConvergence(OptimizationEngineWrapper):
    """
    Wrapper class for self-consistent convergence engine.

    This engine performs convergence studies on multiple input parameters in a
    self-consistent manner. Within each SCF iteration, it converges ALL parameters
    in parallel (each varying independently while others are fixed at their current
    best values), then checks for self-consistency between iterations. The cycle
    continues until all parameters reach self-consistency.

    Parameters
    ----------
    input_values : dict
        Dictionary mapping parameter names to lists of input values to test
    tol : dict
        Dictionary mapping parameter names to convergence tolerances for each parameter
    scf_thr : dict
        Dictionary mapping parameter names to self-consistency thresholds
    result_key : str
        Name of the output/result key which is the value to converge
    convergence_window : dict or int
        Dictionary mapping parameter names to convergence window sizes, or a single
        integer to use the same window size for all parameters (default: 2)
    array_name : str or None
        Name of array within output ArrayData (only needed if output is ArrayData)
    max_scf_iterations : int
        Maximum number of self-consistent field iterations (default: 10)
    logger : optional
        Logger instance for debugging

    Example
    -------
    >>> engine = SelfConsistentConvergence(
    ...     input_values={'kpoints': [2, 4, 6, 8], 'cutoff': [20, 30, 40, 50]},
    ...     tol={'kpoints': 0.01, 'cutoff': 0.01},
    ...     scf_thr={'kpoints': 0.001, 'cutoff': 0.001},
    ...     result_key='energy',
    ...     convergence_window={'kpoints': 3, 'cutoff': 2}
    ... )

    Notes
    -----
    The parallel convergence approach in each SCF iteration means:
    - All parameters are converged simultaneously (not sequentially)
    - Each parameter varies independently while others stay fixed
    - All convergence studies run in parallel for maximum efficiency
    - After all parameters converge, SCF consistency is checked
    - If not SCF-converged, a new iteration begins with updated reference values
    """

    _IMPL_CLASS = _SelfConsistentConvergenceImpl

    def __new__(  # type: ignore  # pylint: disable=too-many-arguments,arguments-differ
        cls,
        input_values: ty.Dict[str, ty.List[ty.Any]],
        tol: ty.Dict[str, float],
        scf_thr: ty.Dict[str, float],
        result_key: str,
        convergence_window: ty.Union[ty.Dict[str, int], int] = 2,
        array_name: ty.Optional[str] = None,
        max_scf_iterations: int = 10,
        logger: ty.Optional[ty.Any] = None,
    ) -> _SelfConsistentConvergenceImpl:
        param_order = list(input_values.keys())

        # Convert convergence_window to dict if it's an int
        if isinstance(convergence_window, int):
            convergence_window_dict = {
                param: convergence_window for param in param_order
            }
        else:
            convergence_window_dict = convergence_window

        return cls._IMPL_CLASS(
            input_values=input_values,
            tol=tol,
            scf_thr=scf_thr,
            result_key=result_key,
            convergence_window=convergence_window_dict,
            array_name=array_name,
            max_scf_iterations=max_scf_iterations,
            current_scf_iteration=0,
            param_order=param_order,
            converged_values={},
            current_indices={param: 0 for param in param_order},
            result_values={param: [] for param in param_order},
            scf_history=[],
            params_converged={param: False for param in param_order},
            initialized=False,
            logger=logger,
        )
