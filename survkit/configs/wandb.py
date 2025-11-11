from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class WandbConfig:
    """Configuration for Wandb logging."""
    # NB: any field without a default value specified or Optional will be required
    name: Optional[str] = field(
        default=None, metadata={"help": "Name for the Weights & Biases run."})
    project: str = field(
        default='survival-moe',
        metadata={"help": "Project name in Weights & Biases."})
    entity: str = field(
        default='tm3229',
        metadata={"help": "Entity (user or team) in Weights & Biases."})
    group: str = field(
        default="testing",
        metadata={"help": "(Experiment) group name in Weights & Biases."})
    dir: str = field(
        default=".",
        metadata={
            "help":
            "Directory for Weights & Biases files. If it doesn't exist wandb will use /tmp."
        })
    mode: str = field(default="online",
                      metadata={
                          "help":
                          "Mode for Weights & Biases runs: {online, offline, disabled}."
                      })
    job_type: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Job type for Weights & Biases runs: {train, eval, test, predict}."
        })
    tags: Optional[List[str]] = field(
        default=None, metadata={"help": "Tags for the Weights & Biases run."})
    notes: Optional[str] = field(
        default=None, metadata={"help": "Notes for the Weights & Biases run."})