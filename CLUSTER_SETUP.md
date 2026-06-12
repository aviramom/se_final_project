# Cluster Setup Guide

**Goal:** run fmeval evaluation jobs from your Mac on `slurm.bgu.ac.il`
via `SlurmRunner` over SSH.

**Your details:**
- Cluster: `slurm.bgu.ac.il`
- Username: `aviramom`
- Local project: `/Users/omeraviram/Projects/final_project`

---

## Step 1 — Verify or create an SSH key on your Mac

Open a terminal on your Mac and run:

```bash
ls ~/.ssh/id_rsa ~/.ssh/id_ed25519 2>/dev/null
```

- If either file is listed, you already have a key — skip to Step 2.
- If nothing is listed, create one:

```bash
ssh-keygen -t ed25519 -C "fmeval-cluster" -f ~/.ssh/id_ed25519
# Press Enter twice to skip the passphrase (easier for automation)
```

---

## Step 2 — Copy your SSH key to the cluster

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub aviramom@slurm.bgu.ac.il
```

You will be asked for your cluster password **once**. After this step, all future
connections are key-based — no password prompt.

**Verify it works:**

```bash
ssh aviramom@slurm.bgu.ac.il "echo SSH OK"
```

You should see `SSH OK` with no password prompt.

---

## Step 3 — Create the fmeval working directory on the cluster

```bash
ssh aviramom@slurm.bgu.ac.il "mkdir -p ~/fmeval_project ~/fmeval_jobs"
```

- `~/fmeval_project` — where the code will live.
- `~/fmeval_jobs`    — where `SlurmRunner` creates per-job directories.

---

## Step 4 — Copy the project to the cluster

Run this from your Mac (it uploads the whole project, excluding data and caches):

```bash
rsync -av --progress \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='data/' \
  --exclude='fmeval.egg-info' \
  /Users/omeraviram/Projects/final_project/ \
  aviramom@slurm.bgu.ac.il:~/fmeval_project/
```

> **Tip:** run the same `rsync` command any time you change the code locally —
> it will only upload the changed files.

---

## Step 5 — Set up a Python environment on the cluster

SSH into the cluster:

```bash
ssh aviramom@slurm.bgu.ac.il
```

Then, inside the cluster session:

```bash
# Check what Python versions are available
module avail python 2>&1 | head -20

# Load a Python 3.11+ module (adjust the exact name to what your cluster has)
module load python/3.11   # or: module load anaconda3, etc.

# Create a virtual environment
python -m venv ~/fmeval_project/.venv

# Activate it
source ~/fmeval_project/.venv/bin/activate

# Install fmeval + all dependencies
pip install --upgrade pip
pip install -r ~/fmeval_project/requirements.txt
pip install -e ~/fmeval_project   # editable install so imports work

# Verify
python -c "import fmeval; print('fmeval import OK')"
```

Note the **full path** to the Python interpreter — you will need it in Step 7:

```bash
which python   # something like /home/aviramom/fmeval_project/.venv/bin/python
```

Exit the cluster session:

```bash
exit
```

---

## Step 6 — Find the right Slurm partition

SSH in and check what partitions are available to you:

```bash
ssh aviramom@slurm.bgu.ac.il "sinfo -o '%P %a %l %D %t' | head -20"
```

Pick a partition you have access to (e.g. `gpu`, `research`, `short`).
If you are not sure, run `squeue` to see what partition your past jobs used.

---

## Step 7 — Run the smoke test from your Mac

Back on your Mac, in the project directory, open a Python session or create a
small script to verify end-to-end submission works:

```bash
cd /Users/omeraviram/Projects/final_project
.venv/bin/python - <<'EOF'
import time
from datetime import datetime

from fmeval.execution import SlurmConfig, SlurmRunner
from fmeval.execution.job import EvaluationJob, JobStatus
from fmeval.config.benchmark_registry import build_default_benchmark_registry
from fmeval.config.model_registry import build_default_model_registry

# ── Configure the runner ──────────────────────────────────────────────────────
cfg = SlurmConfig(
    host="slurm.bgu.ac.il",
    user="aviramom",
    remote_work_dir="/home/aviramom/fmeval_jobs",
    ssh_key_path="/Users/omeraviram/.ssh/id_ed25519",   # adjust if you used id_rsa
    partition="gpu",                                     # adjust to your partition
    time_limit="00:15:00",
    gpus_per_node=0,       # 0 = CPU-only for this smoke test
    cpus_per_task=2,
    mem_gb=8,
    python_bin="/home/aviramom/fmeval_project/.venv/bin/python",
    fmeval_dir="/home/aviramom/fmeval_project",
    env_setup_commands=["module load python/3.11"],      # adjust to your module name
)

runner = SlurmRunner(cfg)

# ── Build a tiny job ─────────────────────────────────────────────────────────
benchmark_registry = build_default_benchmark_registry()
model_registry = build_default_model_registry()

dataset = benchmark_registry.get("tsexam1", max_samples=20)
model   = model_registry.get("mock_always_a")

job = EvaluationJob(
    job_id="smoke-test-001",
    model_name=model.model_name,
    benchmark_name=dataset.name,
    modality="multimodal",
    status=JobStatus.QUEUED,
    created_at=datetime.now(),
    max_samples=20,
)

# ── Submit ────────────────────────────────────────────────────────────────────
print("Submitting job...")
job.handle = runner.submit(job, dataset, model)
print(f"Slurm job ID: {job.handle.slurm_job_id}")

# ── Poll until done (or 10 minutes) ──────────────────────────────────────────
for i in range(60):
    status = runner.get_status(job)
    print(f"[{i*10:3d}s] Status: {status}")
    if status in (JobStatus.COMPLETED, JobStatus.FAILED):
        break
    time.sleep(10)

# ── Fetch the result ─────────────────────────────────────────────────────────
if status == JobStatus.COMPLETED:
    result = runner.get_result(job)
    print("\n" + result.summary())
else:
    print(f"\nJob ended with status: {status}")
    print(f"Check logs at: {job.handle.remote_job_dir}/slurm_*.out")
EOF
```

A successful run ends with a metrics table printed in your terminal.

---

## Step 8 — If the job fails: read the logs

The sbatch script writes stdout and stderr to the job directory on the cluster.
To inspect them from your Mac:

```bash
# List files in the job directory
ssh aviramom@slurm.bgu.ac.il "ls ~/fmeval_jobs/smoke-test-001/"

# Print the stdout log
ssh aviramom@slurm.bgu.ac.il "cat ~/fmeval_jobs/smoke-test-001/slurm_*.out"

# Print the stderr log
ssh aviramom@slurm.bgu.ac.il "cat ~/fmeval_jobs/smoke-test-001/slurm_*.err"
```

Common failure causes and fixes:

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: fmeval` | fmeval not installed on cluster | Re-run `pip install -e ~/fmeval_project` on the cluster |
| `module: command not found` | Wrong `env_setup_commands` | Check the module name with `module avail python` |
| `sbatch: error: ...` | Wrong partition name | Check available partitions (`sinfo`) |
| Job stays `QUEUED` forever | No resources on that partition | Try a different partition or remove the partition constraint |
| `SSH command failed` | Key not accepted | Re-run `ssh-copy-id` or check `ssh_key_path` in SlurmConfig |

---

## Quick reference — useful cluster commands

```bash
# Check your running/pending jobs
ssh aviramom@slurm.bgu.ac.il "squeue -u aviramom"

# Cancel a job
ssh aviramom@slurm.bgu.ac.il "scancel <job_id>"

# Check cluster partitions
ssh aviramom@slurm.bgu.ac.il "sinfo"

# Re-sync code after local changes
rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='.git' --exclude='data/' --exclude='fmeval.egg-info' \
  /Users/omeraviram/Projects/final_project/ \
  aviramom@slurm.bgu.ac.il:~/fmeval_project/
```

---

## Once the smoke test passes

Come back to a Claude Code session in `/Users/omeraviram/Projects/final_project`
and we will:
1. Wire `SlurmRunner` into `EvaluationService` so the UI can submit real jobs.
2. Add real model wrappers (HuggingFace / vLLM).
3. Add more benchmarks with their cluster dataset paths.
