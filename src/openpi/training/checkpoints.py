import concurrent.futures as futures
import dataclasses
import heapq
import json
import logging
import os
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def __init__(self):
        self._executor = futures.ThreadPoolExecutor(max_workers=1)

    def close(self):
        self._executor.shutdown()

    def save(self, directory: epath.Path, args: "CallbackSave"):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: "CallbackSave") -> list[futures.Future]:
        return [self._executor.submit(self.save, directory, args)]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])


class BestCheckpointTracker:
    """Tracks the best checkpoints based on loss values."""
    
    def __init__(self, checkpoint_dir: epath.Path, max_checkpoints: int = 3):
        """Initialize the checkpoint tracker.
        
        Args:
            checkpoint_dir: Directory where checkpoints are stored.
            max_checkpoints: Maximum number of checkpoints to keep.
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.tracker_file = checkpoint_dir / "best_checkpoints.json"
        self.best_checkpoints = []  # List of (loss, step) tuples
        
        # Load existing tracker if it exists
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, "r") as f:
                    data = json.load(f)
                self.best_checkpoints = [(entry["loss"], entry["step"]) for entry in data["checkpoints"]]
                # Convert to a heap for easy management
                heapq.heapify(self.best_checkpoints)
            except Exception as e:
                logging.warning(f"Failed to load checkpoint tracker: {e}")
                self.best_checkpoints = []
    
    def should_save(self, step: int, loss: float) -> bool:
        """Determine if the current checkpoint should be saved.
        
        Args:
            step: Current training step.
            loss: Current loss value.
            
        Returns:
            True if the checkpoint should be saved, False otherwise.
        """
        # Always save if we have fewer than max_checkpoints
        if len(self.best_checkpoints) < self.max_checkpoints:
            return True
        
        # Save if the current loss is better than the worst saved checkpoint
        worst_loss = -self.best_checkpoints[0][0] if self.best_checkpoints else float('inf')
        return loss < worst_loss
    
    def update(self, step: int, loss: float) -> int:
        """Update the tracker with a new checkpoint.
        
        Args:
            step: Current training step.
            loss: Current loss value.
            
        Returns:
            Step to remove, or -1 if no checkpoint should be removed.
        """
        # If we have fewer than max_checkpoints, just add the new one
        if len(self.best_checkpoints) < self.max_checkpoints:
            heapq.heappush(self.best_checkpoints, (-loss, step))
            self._save_tracker()
            return -1
        
        # If the current loss is better than the worst saved checkpoint, replace it
        worst_loss, worst_step = self.best_checkpoints[0]
        if loss < -worst_loss:
            heapq.heapreplace(self.best_checkpoints, (-loss, step))
            self._save_tracker()
            return worst_step
        
        return -1
    
    def _save_tracker(self):
        """Save the tracker to disk."""
        data = {
            "checkpoints": [
                {"loss": -loss, "step": step} 
                for loss, step in self.best_checkpoints
            ]
        }
        with open(self.tracker_file, "w") as f:
            json.dump(data, f, indent=2)


def save_best_checkpoints(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    loss: float,
    max_checkpoints: int = 3,
):
    """Save checkpoint if it's among the best based on loss.
    
    Args:
        checkpoint_manager: Orbax checkpoint manager.
        state: Current training state.
        data_loader: Data loader.
        step: Current training step.
        loss: Current loss value.
        max_checkpoints: Maximum number of checkpoints to keep.
    """
    checkpoint_dir = epath.Path(checkpoint_manager._directory)
    tracker = BestCheckpointTracker(checkpoint_dir, max_checkpoints)
    
    # Check if we should save this checkpoint
    if tracker.should_save(step, loss):
        # Save the current checkpoint
        save_state(checkpoint_manager, state, data_loader, step)
        
        # Update the tracker and potentially remove the worst checkpoint
        step_to_remove = tracker.update(step, loss)
        
        # Remove the worst checkpoint if needed
        if step_to_remove >= 0 and step_to_remove != step:
            try:
                checkpoint_manager.delete(step_to_remove)
                logging.info(f"Removed checkpoint at step {step_to_remove} with higher loss")
            except Exception as e:
                logging.warning(f"Failed to remove checkpoint at step {step_to_remove}: {e}")
    else:
        logging.info(f"Skipping checkpoint at step {step} with loss {loss:.4f} as it's not among the best")
