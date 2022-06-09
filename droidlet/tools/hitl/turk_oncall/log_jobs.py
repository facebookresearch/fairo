import os
import boto3
import logging

from droidlet.tools.hitl.data_generator import DataGenerator
from droidlet.tools.hitl.job_listener import JobListener
from droidlet.tools.hitl.task_runner import TaskRunner

ECS_INSTANCE_TIMEOUT = 45
INTERACTION_JOB_POLL_TIME = 30
INTERACTION_LISTENER_POLL_TIME = 30
HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)
S3_BUCKET_NAME = "droidlet-hitl"
S3_ROOT = "s3://droidlet-hitl"
NSP_OUTPUT_FNAME = "nsp_outputs"
ANNOTATED_COMMANDS_FNAME = "nsp_data.txt"

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
s3 = boto3.resource(
    "s3",
    region_name=AWS_DEFAULT_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)
logger = logging.getLogger()
logger.handlers.clear()
logger.setLevel("INFO")
sh = logging.StreamHandler()
sh.setFormatter(log_formatter)
logger.addHandler(sh)

class TaoLogOutputJob(DataGenerator):
    ''' TODO: 
           1. Unzip log and read
           2. Find Traceback
           3. Output traceback to a file
    '''
    def __init__(self, batch_id: int, timeout: float = -1) -> None:
        super().__init__(timeout)
        self._batch_id = batch_id
    
    def run(self) -> None:
        
        return super().run()

class TaoLogListener(JobListener):
    # TODO: check s3 log files
    # TODO: create stat file
    # TODO: init log output job
    
    def __init__(self, batch_id: int, timeout: float = -1) -> None:
        super(TaoLogListener, self).__init__(timeout=timeout)
        self._batch_id = batch_id

    def run(self, runner: TaskRunner) -> None:
        batch_id = self._batch_id

        while not self.check_is_finished(): 
            
            # create a tao log output job
            tlo_job = TaoLogOutputJob(batch_id=batch_id)
            runner.register_data_generators([tlo_job])
            

            # if not self.check_parent_finished():
            #     finished = False
            
            # self.set_finished(finished)

if __name__ == "__main__":
    runner = TaskRunner()

    # test on hard coded batch id
    tao_log_listener = TaoLogListener(batch_id=)
    runner.register_job_listeners([tao_log_listener])

    runner.run()
