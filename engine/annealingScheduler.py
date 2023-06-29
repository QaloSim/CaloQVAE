"""
Scheduler module to perform annealing a given parameter
"""

from CaloQVAE import logging
logger = logging.getLogger(__name__)

class Scheduler:
    def __init__(start_point: float, end_point: float, method: str = "linear", trigger_value, trigger_var_curr_value: float = None):
       logger.info("Setting up the scheduler")
       self.anneal_var: float = start_point
       self.start_point: float = start_point #start point of annealing 
       self.end_point: float = end_point #end point of annealing
       self.method: str = method
       self.trigger_value: float = trigger_value #trigger value current value
       self.trigger_var_curr_value: float = trigger_var_curr_value #trigger value to start annealing

    def __repr__(self) -> str:
        f"Current value: {self.anneal_var}, method: {self.method}, value to trigger: {self.trigger_value}"

    def get_linear_direction(self) -> int:
        direction: int = 1 if self.start_point < self.end_point else -1
        return direction

    def update_trigger_value(self, trigger_var_value: float):
        logger.debug("Updating trigger variable variable")
        self.trigger_var_curr_value = trigger_var_value
    
    def linear_annealing(self) -> int:
        """
        Linear annealing with a given step
        Returns an error code int
        """
        logger.debug("Doing linear annealing")
        direction: int = self.get_linear_direction()

        #Checking if the value is still within the annealing values
        if direction * self.anneal_var < direction * self.start_point or direction * self.anneal_var > direction * self.end_point:
            logger.warning("Value is outside of the annealing bound")
            return -1
        
        if direction * self.trigger_var_curr_value < direction * self.trigger_value:
            logger.warning("Still below trigger threshold")
            return -1
        
        logger.debug("Updating annealing variable.")
        logger.debug(__repr__)

        return 0

        
