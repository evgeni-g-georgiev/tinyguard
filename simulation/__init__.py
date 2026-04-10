
"""Simulation package — force-load all registered implementations.
                                                                                    
Importing the implementation modules here triggers their @register_*                  
decorators, populating the registries before run_simulation.py uses them.             
                                                                                    
Without this file, the registries would be empty when create_X() is                   
called, even though the implementation files exist on disk.    

Note that # noaq: F401 is for the inter, this import is unused but intentional 
"""                                                                                   
                
# Frozen embedders                                                                    
from simulation.inference_models import embedder_identity      # noqa: F401
from simulation.inference_models import embedder_acoustic      # noqa: F401           

# On-device separators                                                                
from simulation.inference_models import separator_identity     # noqa: F401
from simulation.inference_models import separator_svdd         # noqa: F401    
from simulation.inference_models import separator_gmm          # noqa: F401       
                                                                                    
# Topologies
from simulation.node import topology_isolated                  # noqa: F401           
                
# Merge operators                                                                     
from simulation.node import merge_none                         # noqa: F401