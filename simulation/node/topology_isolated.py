"""Isolated topology — no communication between nodes.
                                                                                    
Each node trains and scores independently. Used as the v1 default and                 
as the baseline to measure federation benefit against.                                
"""                                                                                   
                
from simulation.registry import register_topology                                     
from simulation.node.base_topology import BaseTopology
                                                                                    
                
@register_topology("isolated")
class IsolatedTopology(BaseTopology):

    def neighbours(self, node_id: str) -> list[str]:
        return []
                        