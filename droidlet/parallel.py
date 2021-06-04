import traceback
import queue
from typing import Callable, List
import cloudpickle
# you're wondering wtf? why is numpy needed in this file?
# it's a workaround for https://github.com/pytorch/pytorch/issues/37377
import numpy 
from torch import multiprocessing as mp

multiprocessing = mp.get_context("spawn")

class Process(multiprocessing.Process):
    """
    Class which returns child Exceptions to Parent.
    https://stackoverflow.com/a/33599967/4992248
    """

    def __init__(self, *args, **kwargs):
        multiprocessing.Process.__init__(self, *args, **kwargs)
        self._parent_conn, self._child_conn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        try:
            multiprocessing.Process.run(self)
            self._child_conn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))
            raise e

    @property
    def exception(self):
        if self._parent_conn.poll():
            self._exception = self._parent_conn.recv()
        return self._exception

def _runner(_init_fn, init_args, _process_fn, shutdown_event, input_queue, output_queue):
    init_fn = cloudpickle.loads(_init_fn)
    process_fn = cloudpickle.loads(_process_fn)
    initial_state = init_fn(*init_args)

    while not shutdown_event.is_set():
        try:
            process_args = input_queue.get(block=True, timeout=0.1)
            process_args_aug = (initial_state, *process_args)
            print(process_args)
            print(process_args_aug)
            process_return = process_fn(*process_args_aug)
            output_queue.put(process_return)
        except queue.Empty:
            pass


class BackgroundTask:
    def __init__(self, init_fn: Callable, init_args: List, process_fn: Callable):
        self._init_fn = init_fn
        self._init_args = init_args
        self._process_fn = process_fn
        self._send_queue = multiprocessing.Queue()
        self._recv_queue = multiprocessing.Queue()
        self._shutdown_event = multiprocessing.Event()

    def start(self):
        self._process = Process(target=_runner,
                                args=(
                                    cloudpickle.dumps(self._init_fn),
                                    self._init_args,
                                    cloudpickle.dumps(self._process_fn),
                                    self._shutdown_event,
                                    self._send_queue, self._recv_queue
                                ),)
        self._process.daemon = True
        self._process.start()

    def _raise(self):
        if not hasattr(self, "_process"):
            raise RuntimeError("BackgroundTask has not yet been started."
                               " Did you forget to call .start()?")
        if self._process.exception:
            error, _traceback = self._process.exception
            raise ChildProcessError(_traceback)

    def stop(self):
        self._raise()
        self._shutdown_event.set()

    def put(self, *args):
        self._raise()
        self._send_queue.put(args)

    def get(self, block=True, timeout=None):
        self._raise()
        return self._recv_queue.get(block, timeout)

    def get_nowait(self):
        self._raise()
        return self._recv_queue.get_nowait()
