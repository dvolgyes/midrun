"""Run a slow training job for interactive REST and FUSE inspection.

Start it with `uv run --group examples python -m examples.playground.run`.
Open the dashboard URL printed at fit start or append `/docs` for the API.
Press Ctrl+C in the training terminal to stop the run.
Use `--hooks-source-dir` to supply your own baseline scripts.
"""

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar

import click
import lightning.pytorch as pl
import torch
import torch.distributed as dist
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import v2

from midrun import MidrunMixin

LOG_DIR = Path(__file__).resolve().parent / "logs"


class PlaygroundModel(MidrunMixin, pl.LightningModule):
    """Train slowly enough to inspect a live run.

    :ivar midrun_tunable_hparams: Step delay that the operator can change during training.
    :ivar net: Classifier for 16 by 16 RGB images.
    """

    midrun_tunable_hparams: ClassVar[dict[str, dict[str, float | str]]] = {
        "hparams.step_delay": {"min": 0.0, "max": 5.0, "label": "Step delay (s)"},
    }

    net: nn.Sequential

    def __init__(
        self,
        lr: float = 1e-3,
        step_delay: float = 0.3,
        hooks_source_dir: Path | None = None,
    ) -> None:
        super().__init__(hooks_source_dir=hooks_source_dir)
        self.save_hyperparameters({"lr": lr, "step_delay": step_delay})
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(3 * 16 * 16, 32), nn.ReLU(), nn.Linear(32, 10)
        )

    def on_fit_start(self) -> None:
        """Reports the addresses selected during setup."""
        super().on_fit_start()
        if not self.trainer.is_global_zero:
            return
        if url := self.midrun.rest_url:
            logger.info("Dashboard: {}/  API docs: {}/docs", url, url)
        else:
            logger.warning("The REST dashboard is unavailable.")
        logger.info("FUSE mount: {}", self.midrun.fuse_mountpoint or "unavailable")
        logger.info("Artifacts: {}", self.midrun.state.read().artifacts_path)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Computes class logits for a batch of flattened images.

        Args:
            images: Batch of shape (N, 3, 16, 16).

        Returns:
            Class logits of shape (N, 10).
        """
        logits: torch.Tensor = self.net(images)
        return logits

    def training_step(  # pylint: disable=arguments-differ
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Sleeps briefly, then computes and logs the cross-entropy loss for one batch.

        Args:
            batch: Pair of (images, labels) tensors.
            batch_idx: Index of the batch within the current epoch; unused.

        Returns:
            The training loss to backpropagate.
        """
        del batch_idx
        time.sleep(self.hparams["step_delay"])
        images, labels = batch
        loss = nn.functional.cross_entropy(self(images), labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Builds the Adam optimizer.

        Returns:
            The optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])


@contextmanager
def _distributed_run() -> Iterator[None]:
    """Manage the playground's single-rank process group for the debug server."""
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29511")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        yield
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _build_loader() -> DataLoader[tuple[torch.Tensor, int]]:
    """Builds a small synthetic image-classification dataloader.

    Returns:
        A DataLoader of 16x16 RGB images against 10 fake classes.
    """
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    dataset = FakeData(
        size=64, image_size=(3, 16, 16), num_classes=10, transform=transform
    )
    return DataLoader(dataset, batch_size=8, num_workers=0)


@click.command()
@click.option(
    "--hooks-source-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="Baseline hooks/ directory (light/, heavy/, supporting files) copied into "
    "this run's live hooks tree; see the README's 'Shell hooks' section.",
)
def main(hooks_source_dir: Path | None) -> None:
    """Train in the foreground and report the selected dashboard URL."""
    with _distributed_run():
        tb_logger = TensorBoardLogger(save_dir=str(LOG_DIR), name="playground")
        model = PlaygroundModel(hooks_source_dir=hooks_source_dir)
        trainer = pl.Trainer(
            max_epochs=-1,
            max_time="00:01:00:00",
            logger=tb_logger,
            enable_checkpointing=False,
            accelerator="cpu",
        )
        trainer.fit(model, train_dataloaders=_build_loader())
    logger.info("Stopped.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter  # click supplies args from argv at runtime
