# Imports necessary libraries
import os
import sys
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from subprocess import Popen, PIPE

script_path = os.path.dirname(os.path.realpath(__file__))
feedback = os.path.join(script_path, 'data/feedback.csv')

rerun = [sys.executable, 'collaborative.py'] 

class Monitor(FileSystemEventHandler):
    def __init__(self):
        self.process = None

    def on_modified(self, event):
        if os.path.abspath(event.src_path) == feedback:
            self.rerun_script()

    def on_deleted(self, event):
        if os.path.abspath(event.src_path) == feedback:
            self.rerun_script()

    def rerun_script(self):
        if self.process is not None and self.process.poll() is None:
            print("The collaborative model is still running. Skipping this modification...")
        else:
            print("The feedback file has been modified. The collaborative model is now rerunning ...")
            self.process = Popen(rerun)

if __name__ == "__main__":
    event_handler = Monitor()
    observer = Observer()
    observer.schedule(event_handler, path=script_path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
