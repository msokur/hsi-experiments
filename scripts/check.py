import sys
import os

NODES = int(os.environ.get("SLURM_JOB_NUM_NODES"))
MEM_PER_NODE = int(os.environ.get("SLURM_MEM_PER_NODE"))
TASKS_PER_NODE = int(os.environ.get("SLURM_NTASKS_PER_NODE"))
TOTAL_TASKS = int(os.environ.get("SLURM_NTASKS"))
CPUS_PER_TASK = int(os.environ.get("SLURM_CPUS_PER_TASK"))
CPU_PER_NODE = int(os.environ.get("SLURM_CPUS_ON_NODE"))
MEM_PER_CPU = int(os.environ.get("SLURM_MEM_PER_CPU"))

print("----- SLURM INFORMATION -----")
print(f"Nodes: {NODES} -> Memory per Node: {MEM_PER_NODE} MB")
print(f"Tasks per node: {TASKS_PER_NODE}")
print(f"Total tasks: {TOTAL_TASKS}")
print(f"CPUs per node: {CPU_PER_NODE} -> Memory per CPU: {MEM_PER_CPU} MB")
print(f"CPU per task: {CPUS_PER_TASK} -> Memory per task: {CPUS_PER_TASK * MEM_PER_CPU} MB")
print("-----------------------------")

print(sys.version)

print(sys.path)

print("Check finish!\n")
