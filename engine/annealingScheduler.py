"""
Scheduler module to perform annealing a given parameter
"""

from CaloQVAE import logging
logger = logging.getLogger(__name__)

class Scheduler:
    def __init__(start_point: float, end_point: float, method: str = "linear", trigger_value, trigger_var_curr_value: float = None):
       logger.info("Setting up the scheduler")
       self.start_poing: float = start_point
       self.end_point: float = end_point
       self.method: str = method
       self.trigger_value: float = trigger_value
       self.trigger_var_curr_value: float = trigger_var_curr_value

    def update_trigger_value(trigger_var_value):
        logger.debug("Updating trigger variable variable")
        self.trigger_var_value = trigger_var_value
    
