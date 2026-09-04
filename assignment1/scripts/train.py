from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from huggingface_hub import snapshot_download
from omegaconf import DictConfig, OmegaConf

from cs336_basics.Checkpoint import save_checkpoint
from cs336_basics.Data import Loader
from cs336_basics.Module import AdamW, CrossEntropyLoss, TransformerLM


def ensure_training_dataset(data_config: DictConfig) -> Path:
    dataset_path = Path(data_config.dataset_path)
    tokenized_path = dataset_path / data_config.tokenized_filename
    if tokenized_path.is_file():
        return tokenized_path

    snapshot_download(
        repo_id=data_config.huggingface.processed_repo_id,
        repo_type=data_config.huggingface.repo_type,
        local_dir=dataset_path,
    )
    if not tokenized_path.is_file():
        raise FileNotFoundError(
            f"Expected tokenized dataset at {tokenized_path} after downloading "
            f"{data_config.huggingface.processed_repo_id}."
        )

    return tokenized_path


def initialize_wandb(config: DictConfig):
    wandb_config = config.training.wandb
    if not wandb_config.enabled:
        return None

    return wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        name=wandb_config.name,
        mode=wandb_config.mode,
        dir=config.training.log_dir,
        tags=list(wandb_config.tags),
        config=OmegaConf.to_container(config, resolve=True),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    data_config = config.data
    model_config = config.model
    training_config = config.training

    torch.manual_seed(training_config.seed)
    log_dir = Path(training_config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    tokenized_path = ensure_training_dataset(data_config)
    dataset = np.memmap(
        tokenized_path,
        dtype=data_config.dtype,
        mode="r",
    )

    data_loader = Loader(
        dataset=dataset,
        batch_size=training_config.batch_size,
        context_length=model_config.context_length,
        device=training_config.device,
    )

    model = TransformerLM(
        vocab_size=model_config.vocab_size,
        context_length=model_config.context_length,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        d_model=model_config.d_model,
        d_ff=model_config.d_ff,
        rope_theta=model_config.rope_theta,
    )

    model = model.to(training_config.device)
    model.train()
    loss_fn = CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        betas=tuple(training_config.betas),
        eps=training_config.optimizer_eps,
    )

    run = initialize_wandb(config)
    try:
        step = -1
        for step, (x, y) in enumerate(data_loader.get_batch_iterable()):
            optimizer.zero_grad()

            logits = model(x)  # (batch_size, context_length, vocab_size)
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            loss.backward()
            optimizer.step()

            if step % training_config.log_every == 0:
                loss_value = loss.item()
                print(f"step={step:6d}/{len(data_loader)}, loss={loss_value:.4f}")
                if run is not None:
                    run.log(
                        {
                            "train/loss": loss_value,
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                            "train/progress": step / len(data_loader),
                        },
                        step=step,
                    )

            if step % training_config.save_every == 0:
                save_checkpoint(model, optimizer, step, log_dir / f"model{step}.pt")

        save_checkpoint(model, optimizer, step, log_dir / "model_final.pt")
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()