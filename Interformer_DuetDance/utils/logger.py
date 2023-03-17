import os
import time
import platform


# ----------------------------------------------------------------------------------------------------------------------
class Logger:
    """
    A utility logger that prints everything into the console, and also logs everything printed to a unique text file.
    Very handy if the goal is to spawn many runs especially on different compute nodes.
    """
    def __init__(self):
        self.log_dir = "logs"  # The directory that would contain all the output files
        self.log_path = ""
        self.log_initialized = False
        self.name = None          # Log file's name
        self.dataset_name = None  # Used in the log's filename

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name
        self.name = "{}-{}-{}".format(dataset_name,
                                      time.strftime("%Y%m%d-%H%M%S"),
                                      platform.node())

    def init_log(self):
        if self.dataset_name is None:
            raise Exception("The dataset name is not set! Make sure to call 'set_dataset_name()' before logging.")

        self.log_initialized = True
        self.log_path = os.path.join(self.log_dir, self.name + ".txt")

        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

    def log(self, message_str):
        if not self.log_initialized:
            self.init_log()

        with open(self.log_path, "a") as logfile:
            logfile.write(message_str + "\n")
            print(message_str)

    def log_dataset(self, dataset):
        if not self.log_initialized:
            self.init_log()

        # Logs a dataset, and the hyperparameters
        if dataset is not None:
            with open(self.log_path, "a") as f:
                f.writelines(str(dataset))

    def __call__(self, message_str):
        self.log(message_str)


# ----------------------------------------------------------------------------------------------------------------------
# The singleton instance
log = Logger()
