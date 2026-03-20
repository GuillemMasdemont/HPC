# Arnes HPC Cluster Guide

## Connection and Authentication
To connect to the Arnes cluster, you must use an SSH key followed by two-factor authentication (OTP).

* **Initial Verification:** ```bash
    ssh gm64359@hpc-otp.arnes.si
    ```
    *This generates a verification code for your session.*

* **Cluster Access:**
    Use the same command to log in once the code is verified:
    ```bash
    ssh gm64359@hpc-login1.arnes.si
    ```
Then you will be asked to enter the authentification key, which is available in your phone app. 

---

## Cluster Architecture
When you log in, you are interacting with the **Login Node**. The actual heavy lifting happens behind the scenes:

* **Master Node:** Hidden from users; manages the entire system.
* **Compute Nodes:** Approximately 100 high-performance computers where your code actually runs.
* **Networking:** A dedicated fast Ethernet/interconnect network links all compute nodes together for high-speed communication.
* **Storage Nodes:** Houses all hard drives and data. While the storage is centralized, users primarily interact with the compute nodes.



---

## Job Scheduling with SLURM
**SLURM** (Simple Linux Utility for Resource Management) is the industry standard (used in ~90% of HPC systems) for scheduling jobs and distributing them across the compute nodes.

### Essential Commands
To view the status of the cluster and see the specific node list, use:
```bash
sinfo --long --Node
```

# Arnes HPC: Node States, SLURM, and Job Management

## 1. Monitoring Node States
When running `sinfo --long --Node`, the **State** column is crucial for understanding availability:

* **IDLE:** The node is completely free and ready for jobs.
* **MIXED:** The node is running jobs but still has available resources (CPU/RAM).
* **ALLOCATED:** The node is at full capacity and cannot accept new jobs.

### Understanding Node Specifications
* **CPUS:** Total number of cores available on the node.
* **S:C:T (Sockets:Cores:Threads):** * *Example:* 2:32:2 means 2 Sockets, 32 Cores per socket, and 2 Threads per core.
    * **Formula:** $Total\ CPUs = S \times C \times T$
* **Memory:** Total RAM available (usually displayed in MB, in the provided case we saw 256 GB).
* **TMP_disk:** Temporary local storage attached to the specific node.
* **Weight:** A value used by the SLURM scheduler to prioritize which nodes to use first.
* **Reason:** If a node is down, the reason appears here; "none" means the node is healthy and online.

---

## 2. Managing Jobs with `squeue`
To see what is happening in the cluster, use `squeue`.

* **ID:** The unique job number.
* **ST (State):** * `PD` (Pending): Waiting for resources.
    * `R` (Running): Currently executing.
* **Time:** How long the job has been running.

### Useful Filters:

## 1. Monitoring Jobs (`squeue`)
Use `squeue` to track your jobs in the cluster.

| Command | Description |
| :--- | :--- |
| `squeue --me` | Show all your jobs. |
| `squeue --me --state=PD` | Show your pending (queued) jobs. |
| `squeue --me --state=R` | Show your currently running jobs. |

---

## 2. Running Interactive Jobs (`srun`)
The `srun` command submits a task to the scheduler, which then allocates a compute node.

### Basic Usage
```bash
hostname          # Runs on the login node
srun hostname     # Runs on an allocated compute node (e.g., wn012.arnes.si)
```

## 2. Inspecting the System (`scontrol`)
While administrators use `scontrol` to modify settings, users use it to view detailed hardware and partition specifications.

| Command | Purpose |
| :--- | :--- |
| `scontrol show partition` | Shows job constraints, time limits, and allowed nodes for every partition. |
| `scontrol show node <name>` | Shows specific hardware (e.g., `wn201`). |

### Hardware Nuances
* **GPU Memory:** Be specific with node selection if needed. For example, a **Tesla V100** might have **32GB**, while a newer version could have **80GB** of RAM.

---

## 3. Batch Scripts (`.sh`)
For most work, you should use **Batch Scripts** rather than interactive commands. 

### Workflow:
1.  **Create:** Write a `.sh` file. The top of the file contains `#SBATCH` directives to define resources (CPU, RAM, Time).
2.  **Submit:**
    ```bash
    sbatch job.sh
    ```
3.  **Monitor:** Check status with `squeue --me`.
4.  **Results:** Output is automatically saved to a log file (e.g., `slurm-jobID.out` or your custom `.log`). View it with:
    ```bash
    cat job-output.log
    ```



---

## 4. Terminating Jobs
If a job is stuck or no longer needed:
* **Specific Job:** `scancel <JOB_ID>`
* **All Your Jobs:** `scancel --me`

---

## ⚠️ Important Constraints
* **Login Node:** Treat this as a "lobby" only. **Do not run heavy computations here.**
* **Storage & Small Files:** All files in your login directory live on a **Storage Node** accessible by all compute nodes.
* **Performance Tip:** Avoid working with thousands of small files. Distributed filesystems (like Lustre or NFS) are optimized for large, sequential data. High file counts will significantly slow down your I/O performance.


