# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Multi-task Early Stopping ^^^^^^^^^^^^^^

Monitor metrics and stop training when they stop improving.

"""
import logging
import dataclasses
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

import lightning.pytorch as pl
from lightning.fabric.utilities.rank_zero import _get_rank
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_prefixed_message, rank_zero_warn

log = logging.getLogger(__name__)


@dataclasses.dataclass
class MonitorInformation():
    monitor: str
    min_delta: float
    patience: int
    mode: str
    strict: str
    check_finite: bool
    stopping_threshold: float
    divergence_threshold: float
    wait_count: float
    monitor_op: Callable
    best_score: float


class MultitaskEarlyStopping(Callback):
    r"""Monitor metrics and stop training when they stop improving.

    Args:
        monitor: quantities to be monitored.
        min_delta: minimum change in the monitored quantities to qualify as an improvement, i.e. an absolute
            change of less than or equal to `min_delta`, will count as no improvement.
        patience: number of checks with no improvement
            after which training will be stopped. Under the default configuration, one check happens after
            every training epoch. However, the frequency of validation can be modified by setting various
            parameters on the ``Trainer``, for example ``check_val_every_n_epoch`` and ``val_check_interval``.

            .. note::

                It must be noted that the patience parameter counts the number of validation checks with
                no improvement, and not the number of training epochs. Therefore, with parameters
                ``check_val_every_n_epoch=10`` and ``patience=3``, the trainer will perform at least 40 training
                epochs before being stopped.

        verbose: verbosity mode.
        mode: one of ``'min'``, ``'max'``. In ``'min'`` mode, training will stop when the quantities
            monitored have stopped decreasing and in ``'max'`` mode it will stop when the quantities
            monitored have stopped increasing.
        strict: whether to crash the training if `monitor` is not found in the validation metrics.
        check_finite: When set ``True``, stops training when the monitor becomes NaN or infinite.
        stopping_threshold: Stop training immediately once the monitored quantities reach this threshold.
        divergence_threshold: Stop training as soon as the monitored quantities becomes worse than this threshold.
        check_on_train_epoch_end: whether to run early stopping at the end of the training epoch.
            If this is ``False``, then the check runs at the end of the validation.
        log_rank_zero_only: When set ``True``, logs the status of the early stopping callback only for rank 0 process.
        stopping_mode: Whether to check all metrics for improvement (all) or if any metric improvement (any)
            suffices to continue training.

    Raises:
        MisconfigurationException:
            If ``mode`` is none of ``"min"`` or ``"max"``.
        RuntimeError:
            If the metric ``monitor`` is not available.

    Example::

        >>> from lightning.pytorch import Trainer
        >>> from lightning.pytorch.callbacks import MultitaskEarlyStopping
        >>> early_stopping = MultitaskEarlyStopping('val_loss', 'val_aux_loss')
        >>> trainer = Trainer(callbacks=[early_stopping])

    """
    mode_dict = {"min": torch.lt, "max": torch.gt}
    stopping_modes = ["all", "any"]

    order_dict = {"min": "<", "max": ">"}

    def __init__(
        self,
        monitor: Union[str, list[str]],
        min_delta: Union[float, list[float]] = 0.0,
        patience: Union[int, list[int]] = 3,
        verbose: bool = False,
        mode: Union[str, list[str]] = "min",
        strict: bool = True,
        check_finite: Union[bool, list[bool]] = True,
        stopping_threshold: Optional[Union[float, list[float]]] = None,
        divergence_threshold: Optional[Union[float, list[float]]] = None,
        check_on_train_epoch_end: Optional[bool] = None,
        log_rank_zero_only: bool = False,
        stopping_mode: str = "all",
    ):
        super().__init__()
        monitor = monitor if isinstance(monitor, list) else [monitor]

        min_delta = min_delta if isinstance(min_delta, list) else [min_delta]
        if len(min_delta) != len(monitor):
            min_delta *= len(monitor)

        patience = patience if isinstance(patience, list) else [patience]
        if len(patience) == 1 and len(patience) != len(monitor):
            patience *= len(monitor)

        mode = mode if isinstance(mode, list) else [mode]
        if len(mode) == 1 and len(mode) != len(monitor):
            mode *= len(monitor)

        strict = strict if isinstance(strict, list) else [strict]
        if len(strict) == 1 and len(strict) != len(monitor):
            strict *= len(monitor)

        check_finite = check_finite if isinstance(check_finite, list) else [check_finite]
        if len(check_finite) == 1 and len(check_finite) != len(monitor):
            check_finite *= len(monitor)

        stopping_threshold = stopping_threshold if isinstance(stopping_threshold, list) else [stopping_threshold]
        if len(stopping_threshold) == 1 and len(stopping_threshold) != len(monitor):
            stopping_threshold *= len(monitor)
            
        divergence_threshold = \
            divergence_threshold if isinstance(divergence_threshold, list) else [divergence_threshold]
        if len(divergence_threshold) == 1 and len(divergence_threshold) != len(monitor):
            divergence_threshold *= len(monitor)

        wait_count = [0 for _ in range(len(monitor))]

        torch_inf = torch.tensor(torch.inf)
        self.monitor_dict = {}
        self.should_stop_previously = []
        for _monitor, _min_delta, _patience, _mode, _strict, _check_finite, \
            _stopping_threshold, _divergence_threshold, _wait_count in zip(
                monitor, min_delta, patience, mode, strict, check_finite,
                stopping_threshold, divergence_threshold, wait_count):

            if _mode not in self.mode_dict:
                raise MisconfigurationException(f"`mode` can be {', '.join(self.mode_dict.keys())}, got {_mode}")
            _monitor_op = self.mode_dict[_mode]
            _min_delta *= 1 if _monitor_op == torch.gt else -1
            _best_score = torch_inf if _monitor_op == torch.lt else -torch_inf

            monitor_info = MonitorInformation(
                _monitor, _min_delta, _patience, _mode, _strict, _check_finite,
                _stopping_threshold, _divergence_threshold, _wait_count, _monitor_op, _best_score)
            self.monitor_dict[_monitor] = monitor_info

        self.verbose = verbose
        self.stopped_epoch = 0
        self._check_on_train_epoch_end = check_on_train_epoch_end
        self.log_rank_zero_only = log_rank_zero_only
        if stopping_mode not in self.stopping_modes:
            raise MisconfigurationException((f"`stopping_mode` can be {', '.join(self.stopping_modes)}, "
                                             f"got {self.stopping_mode}"))
        self.stopping_mode = stopping_mode

    @property
    def state_key(self) -> str:
        return self._generate_state_key(**self.monitor_dict)

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        if self._check_on_train_epoch_end is None:
            # if the user runs validation multiple times per training epoch or multiple training epochs without
            # validation, then we run after validation instead of on train epoch end
            self._check_on_train_epoch_end = trainer.val_check_interval == 1.0 and trainer.check_val_every_n_epoch == 1

    def _validate_condition_metric(self, monitor_info: MonitorInformation, logs: Dict[str, Tensor]) -> bool:
        monitor_val = logs.get(monitor_info.monitor)

        error_msg = (
            f"Early stopping conditioned on metric `{monitor_info.monitor}` which is not available."
            " Pass in or modify your `MultitaskEarlyStopping` callback to use any of the following:"
            f' `{"`, `".join(list(logs.keys()))}`'
        )

        if monitor_val is None:
            if monitor_info.strict:
                raise RuntimeError(error_msg)
            if self.verbose > 0:
                rank_zero_warn(error_msg, category=RuntimeWarning)

            return False

        return True

    def state_dict(self) -> Dict[str, Any]:
        return {
            monitor_key: {
                "wait_count": monitor_info.wait_count,
                "stopped_epoch": monitor_info.stopped_epoch,
                "best_score": monitor_info.best_score,
                "patience": monitor_info.patience,
            } for monitor_key, monitor_info in self.monitor_dict.items()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for metric_key, info_dict in state_dict.items():
            for monitor_key, monitor_info in self.monitor_dict.items():
                if metric_key == monitor_key:
                    monitor_info.wait_count = info_dict["wait_count"]
                    monitor_info.stopped_epoch = info_dict["stopped_epoch"]
                    monitor_info.best_score = info_dict["best_score"]
                    monitor_info.patience = info_dict["patience"]

    def _should_skip_check(self, trainer: "pl.Trainer") -> bool:
        from lightning.pytorch.trainer.states import TrainerFn

        return trainer.state.fn != TrainerFn.FITTING or trainer.sanity_checking

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)

    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        if trainer.fast_dev_run:  # disable early_stopping with fast_dev_run
            return

        should_stops = []
        reasons = []
        skipped_metrics = 0
        for metric, monitor_info in self.monitor_dict.items():
            if not self._validate_condition_metric(monitor_info, logs):
                skipped_metrics += 1
                continue  # short circuit if metric not present

            if monitor_info.monitor in self.should_stop_previously:
                should_stop = True
                reason = (f"Monitored metric {monitor_info.monitor} failed previously. It will contribute to "
                          "the stopping criteria from before.")
            else:
                current = logs[monitor_info.monitor].squeeze()
                should_stop, reason = self._evaluate_stopping_criteria(monitor_info, current)
                should_stops.append(should_stop)
                if should_stop:
                    self.should_stop_previously.append(monitor_info.monitor)
            if reason is not None:
                reasons.append(reason)

        # determine whether to stop based on stopping_mode.
        should_stop = False
        if self.stopping_mode == 'all':
            should_stop = sum(should_stops) == len(self.monitor_dict) - skipped_metrics
        else:  # self.stopping_mode == 'any'
            should_stop = sum(should_stops) > 0

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reasons and self.verbose:
            summary = (f"Training {('was stopped early') if should_stop else 'continues'}."
                       " Individual outputs of metric checking: ")
            self._log_info(trainer, summary + " ; ".join(reasons), self.log_rank_zero_only)

    def _evaluate_stopping_criteria(self, monitor_info, current: Tensor) -> Tuple[bool, Optional[str]]:
        should_stop = False
        reason = None
        if monitor_info.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {monitor_info.monitor} = {current} is not finite."
                f" Previous best value was {monitor_info.best_score:.3f}."
            )
        elif monitor_info.stopping_threshold is not None and monitor_info.monitor_op(
          current, monitor_info.stopping_threshold):
            should_stop = True
            reason = (
                "Stopping threshold reached:"
                f" {monitor_info.monitor} = {current} {self.order_dict[monitor_info.mode]} "
                f"{monitor_info.stopping_threshold}."
            )
        elif monitor_info.divergence_threshold is not None and monitor_info.monitor_op(
          -current, -monitor_info.divergence_threshold):
            should_stop = True
            reason = (
                "Divergence threshold reached:"
                f" {monitor_info.monitor} = {current} {self.order_dict[monitor_info.mode]} "
                f"{monitor_info.divergence_threshold}."
            )
        elif monitor_info.monitor_op(current - monitor_info.min_delta, monitor_info.best_score.to(current.device)):
            should_stop = False
            reason = self._improvement_message(monitor_info, current)
            monitor_info.best_score = current
            monitor_info.wait_count = 0
        else:
            monitor_info.wait_count += 1
            if monitor_info.wait_count >= monitor_info.patience:
                should_stop = True
                reason = (
                    f"Monitored metric {monitor_info.monitor} did not improve in the last {monitor_info.wait_count} "
                    f"records. Best score: {monitor_info.best_score:.3f}."
                )

        return should_stop, reason

    def _improvement_message(self, monitor_info: MonitorInformation, current: Tensor) -> str:
        """Formats a log message that informs the user about an improvement in the monitored score."""
        if torch.isfinite(monitor_info.best_score):
            msg = (
                f"Metric {monitor_info.monitor} improved by {abs(monitor_info.best_score - current):.3f} >="
                f" min_delta = {abs(monitor_info.min_delta)}. New best score: {current:.3f}"
            )
        else:
            msg = f"Metric {monitor_info.monitor} improved. New best score: {current:.3f}"
        return msg

    @staticmethod
    def _log_info(trainer: Optional["pl.Trainer"], message: str, log_rank_zero_only: bool) -> None:
        rank = _get_rank(
            strategy=(trainer.strategy if trainer is not None else None),  # type: ignore[arg-type]
        )
        if trainer is not None and trainer.world_size <= 1:
            rank = None
        message = rank_prefixed_message(message, rank)
        if rank is None or not log_rank_zero_only or rank == 0:
            log.info(message)
