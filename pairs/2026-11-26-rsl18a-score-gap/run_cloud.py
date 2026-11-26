"""Run one single-threaded paired worker per physical core, pinning repeats for each image."""
import collections
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

home=Path.home()
cpu_map=subprocess.check_output(["lscpu","-p=CPU,CORE,SOCKET"],text=True)
cores={}
for line in cpu_map.splitlines():
    if line.startswith("#"):continue
    logical,core,socket=map(int,line.split(","))
    cores.setdefault((socket,core),logical)
cpus=list(cores.values())
out=home/"output"
out.mkdir(exist_ok=True)
assert not list(out.glob("lane-*.csv"))
(out/"cpu-map.txt").write_text(cpu_map)
(out/"cpu-info.txt").write_text(subprocess.check_output(["lscpu"],text=True))
env=dict(os.environ,OPENBLAS_NUM_THREADS="1",JULIA_NUM_THREADS="1",OMP_NUM_THREADS="1",JULIA_PKG_PRECOMPILE_AUTO="0")
def work(lane):
    with (out/f"worker-{lane}.log").open('w') as log:
        subprocess.run(["taskset","-c",str(cpus[lane]),str(home/"julia-1.12.6/bin/julia"),"--project=benchmarks",str(home/"run.jl"),str(out),str(lane),str(len(cpus)),"100","MNIST.RSL18a_linf0.1_authors"],cwd=home/"code",env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
with ThreadPoolExecutor(len(cpus)) as pool:
    list(pool.map(work,range(len(cpus))))
subprocess.run(["python3",str(home/"summarize.py"),str(out)],check=True)
(out/"complete").touch()
