"""
Scheduler module to perform annealing a given parameter
"""

from CaloQVAE import logging
logger = logging.getLogger(__name__)

class Scheduler:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.anneal_var = self.start_point

    def __repr__(self) -> str:
        return f"Current value: {self.anneal_var}, method: {self.method}, value to trigger: {self.trigger_value}"

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
        logger.debug("Generatl annealing function")
        #For the future maybe a map makes more sense
        if self.method == "linear":
            status: int = self.linear_annealing
            return status
        
        else:
            logger.error("Annealing method not supported")
    
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
        logger.debug(self.__repr__)

        self.anneal_var += self.anneal_step * direction

        return 0

        
