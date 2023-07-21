"""
Scheduler module to perform annealing a given parameter
"""

import math
from CaloQVAE import logging
logger = logging.getLogger(__name__)

class Scheduler:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.anneal_var = self.start_point

    def __repr__(self) -> str:
        return f"Current value: {self.anneal_var}, method: {self.method}, step: {self.anneal_step}"

    def get_linear_direction(self) -> int:
        direction: int = 1 if self.start_point < self.end_point else -1
        return direction

    def update_trigger_value(self, trigger_var_value: float):
        logger.debug(f"Updating trigger variable variable to {trigger_var_value}")
        self.trigger_var_curr_value = trigger_var_value

    def get_annealing_var(self) -> float:
        return self.anneal_var

    def anneal(self) -> int:
        """
        General annealing function to steer into the chosen method
        """
        logger.debug("General annealing function")
        #For the future maybe a map makes more sense
        if self.method == "linear":
            status: int = self.linear_annealing()
            return status
        
        else:
            logger.error("Annealing method not supported")
            return -1
    
    def check_ready_annealing(self) -> bool:
        #Checking if the value is still within the annealing values
        direction: int = self.get_linear_direction()

        if direction * self.anneal_var < direction * self.start_point or direction * self.anneal_var > direction * self.end_point:
            logger.warning("Value is outside of the annealing bound")
            return False
        
        #This is ignoring direction of annealing
        if self.trigger_var_curr_value < self.trigger_value:
            logger.warning("Still below trigger threshold")
            return False
 
        return True
    
    def linear_annealing(self) -> int:
        """
        Linear annealing with a given step
        Returns an error code int
        """
        logger.debug("Doing linear annealing")
        logger.debug(self.__repr__)
        direction: int = self.get_linear_direction()

        if self.check_ready_annealing(): 
            self.anneal_var += self.anneal_step * direction
            return 0

        else:
            return -1

    def exponential_annealing(self):
        """
        Method for exponential annealing
        Returns an error int code
        """
        #TODO Add smoothed decay rate
        logger.debug("Doing exponential annealing")
        logger.debug(self.__repr__)
        if self.check_ready_annealing(): self.anneal_var *= math.exp(-self.anneal_step)
        else: return -1

        return 0