from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SlurmConfig:
    """SSH + Slurm configuration for remote job submission.

    host:
        Slurm login node hostname, e.g. "slurm.bgu.ac.il".
    user:
        SSH username on the cluster.
    remote_work_dir:
        Absolute path on the cluster where job directories are created.
        Each job gets its own subdirectory: {remote_work_dir}/{job_id}/
    ssh_key_path:
        Path to the SSH private key file. None = use the SSH agent / default key.
    partition:
        Slurm partition name (--partition). None omits the directive.
    time_limit:
        Wall-clock limit in HH:MM:SS format.
    gpus_per_node:
        Number of GPUs to request. 0 = CPU-only (omits --gres=gpu).
    cpus_per_task:
        Number of CPU cores per task.
    mem_gb:
        Memory in gigabytes.
    python_bin:
        Python interpreter path on the cluster, e.g. "/home/user/.venv/bin/python".
    fmeval_dir:
        If set, prepended to PYTHONPATH so fmeval is importable without a
        cluster-side pip install. Typically the root of the cloned repo on the cluster.
    extra_sbatch_directives:
        Raw #SBATCH lines appended verbatim to the script header,
        e.g. ["#SBATCH --qos=high", "#SBATCH --account=mylab"].
    env_setup_commands:
        Shell commands run before the Python invocation,
        e.g. ["module load cuda/12.1", "conda activate fmeval"].
    """

    host: str
    user: str
    remote_work_dir: str
    ssh_key_path: str | None = None
    partition: str | None = None
    time_limit: str = "02:00:00"
    gpus_per_node: int = 0
    cpus_per_task: int = 4
    mem_gb: int = 16
    python_bin: str = "python"
    fmeval_dir: str | None = None
    extra_sbatch_directives: list[str] = field(default_factory=list)
    env_setup_commands: list[str] = field(default_factory=list)
