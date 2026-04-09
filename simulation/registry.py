"""Component registries for config-driven architecture switching.                     
                                                                                        
  Each swappable component type gets its own registry. Register implementations         
  with the decorator, instantiate by name from YAML config.

  Usage::                                                                               
   
      from simulation.registry import register_embedder                                 
                                                                                      
      @register_embedder("acoustic_encoder")
      class AcousticEmbedder(BaseFrozenEmbedder):
          ...

      # Then in config:  frozen_embedder.name: "acoustic_encoder"                       
      # And in code:     embedder = create_embedder("acoustic_encoder", 
  checkpoint_path=...)                                                                  
""" 

from __future__ import annotations                                                    

from typing import Any                                                                
                                                                                    

def _make_registry(component_name: str) -> tuple:
    """Create a (register, create, list) triple for one component type.

    Args:
        component_name: Human-readable name for error messages.

    Returns:                                                                          
        (register_fn, create_fn, list_fn) tuple.
    """                                                                               
    _registry: dict[str, type] = {}                                                 

    def register(name: str):
        """Decorator that registers a class under *name*."""

        def decorator(cls: type) -> type:                                             
            if name in _registry:
                raise ValueError(                                                     
                    f"{component_name} '{name}' already registered "                
                    f"by {_registry[name].__name__}"
                )
            _registry[name] = cls
            return cls
                                                                                    
        return decorator
                                                                                    
    def create(name: str, **kwargs: Any):                                           
        """Instantiate a registered component by name."""
        if name not in _registry:
            available = ", ".join(sorted(_registry))
            raise ValueError(
                f"Unknown {component_name} '{name}'. Available: [{available}]"
            )                                                                         
        return _registry[name](**kwargs)
                                                                                    
    def list_all() -> list[str]:                                                    
        """Return sorted list of registered names."""
        return sorted(_registry)

    return register, create, list_all    


register_embedder, create_embedder, list_embedders = _make_registry("embedder")     
register_separator, create_separator, list_separators = _make_registry("separator")
register_merge, create_merge, list_merges = _make_registry("merge")
register_topology, create_topology, list_topologies = _make_registry("topology")  