from droidlet.tools.hitl.data_generator import DataGenerator
from droidlet.tools.hitl.job_listener import JobListener
from droidlet.tools.hitl.task_runner import TaskRunner

class OnCallLogOutJob(DataGenerator):
    def __init__(self, timeout: float = -1) -> None:
        super().__init__(timeout)
    
    def run(self) -> None:
        return super().run()

class OnCallLogListener(JobListener):
    # TODO: check s3 log files
    # TODO: create stat file
    # TODO: init log output job
    
    def __init__(self, timeout: float = -1) -> None:
        super().__init__(timeout)

    def run(self) -> None:
        return super().run()
    