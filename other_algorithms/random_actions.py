import numpy as np
import sys
sys.path.append(r"/root/HARL/")

class ra_agent(object):
    
    def __init__(
        self,
    ) -> None:
        pass
        
    def generate_action(
        self,
        max_action : int,
        agent_number : int,
    ):
        actions = []
        for _ in range(agent_number):
            # generate the random values in the range of 0 to 1 of the size of max_action
            action = np.random.rand(max_action)
            actions.append(action)
        return actions



