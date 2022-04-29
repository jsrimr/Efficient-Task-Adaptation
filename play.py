import subprocess
from multiprocessing import Process
import sys
import time


def process(i):
    subprocess.run(
        args=[
            sys.executable, 
            "extract_skill_aspect.py",
            "agent=diayn",
            "task=walker_stand",
            "snapshot_ts=2000000",
            "obs_type=states",
            f"meta={i}"
        ]
    )

procs = []
for i in range(1,16):
    p = Process(target=process, args=(i,))
    p.start()
    procs.append(p)
    time.sleep(5)

for p in procs:
    p.join()

