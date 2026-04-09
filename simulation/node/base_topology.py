"""Base class for network topologies.
                                                                                    
A topology defines which nodes can communicate with each other during
federated learning. Each node queries the topology to discover its                    
neighbours within the same machine type.                                              
"""                                                                                   
                                                                                    
from abc import ABC, abstractmethod                                                   
                
                                                                                    
class BaseTopology(ABC):
                                                                                    
    @abstractmethod
    def neighbours(self, node_id: str) -> list[str]:
        """Return the node IDs that this node can communicate with.

        Args:                                                                         
            node_id: The querying node's identifier (e.g. "fan_id_00").
                                                                                    
        Returns:
            List of neighbour node IDs. Empty list means no communication.            
        """ 