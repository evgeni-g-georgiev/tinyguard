"""Base class for merge operators.

A merge operator defines how a node incorporates state received from its              
neighbours during federation. The node calls merge after receiving
shareable state from all neighbours in its topology.                                  
"""                                                                                   
                                                                                    
from abc import ABC, abstractmethod                                                   
                

class BaseMerge(ABC):

    @abstractmethod
    def merge(self, local_state: dict, neighbour_states: list[dict]) -> dict:
        """Combine local state with neighbour states.                                 

        Args:                                                                         
            local_state: This node's current shareable state.
            neighbour_states: List of states received from neighbours.                

        Returns:                                                                      
            Merged state dict to load back into the local separator.
        """  