"""No-op merge — ignores neighbour state entirely.
                                                                                    
Used when federation is disabled (topology=isolated) or as the solo                   
baseline. Returns local state unchanged.                                              
"""                                                                                   
                
from simulation.registry import register_merge
from simulation.node.base_merge import BaseMerge
                                                                                    

@register_merge("none")                                                               
class NoMerge(BaseMerge):

    def merge(self, local_state: dict, neighbour_states: list[dict]) -> dict:
        return local_state