import sys
sys.path.append("/home/chinmays/Documents/Masters/Classes/ECE542/Competition_Project/ece542-competition-project")
from ml_utils.logger import WandBLogger

config = {"model": "DummyModel"}
logger = WandBLogger("ece542-competition-project", run_name="trial-run", config=config, group="trial-group")

# Do some stuff
for i in range(10):
    logger.log_epoch({"acc": i**2, "loss": i/2}, epoch=i)
logger.summarize_run({"max_acc": 9**2, "min_loss": 9/2})
logger.commit_logs()
