import fbrp

fbrp.process(
    name="proc",
    runtime=fbrp.Conda(
        dependencies=["python>=3.7"],
        run_command=["python3", "proc.py"],
    ),
)

# fbrp.cmd.logs("proc", old=True)
# fbrp.main()

fbrp.cmd.up("proc", reset_logs=True)
import time
time.sleep(1)
fbrp.cmd.down("proc")
fbrp.cmd.logs("proc", old=True)
