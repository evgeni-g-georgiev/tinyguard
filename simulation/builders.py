"""Component construction from YAML config.
                                                                                                    
Reads config selectors and named blocks, instantiates the corresponding
registered components. This is the only module that knows about the                                   
registry and the concrete preprocessor implementations.                                               
"""   

import numpy as np 

from simulation.registry import(
    create_embedder,
    create_separator,
    create_merge,
    create_topology,
)
from simulation.node.node import Node 

def create_from_config(create_fn, config: dict, selector_key: str):
    """Look up a selector and instantiate its named block.                                            
                                                                                                    
    Args:
        create_fn: Registry create function (e.g. create_separator).                                  
        config: Full config dict.                                                                     
        selector_key: Top-level key naming the selector
                    (e.g. "frozen_embedder", "topology").                                             
                                                                                                    
    Returns:
        Instantiated component with the selector's params applied.                                    
    """                                                                                               
    name = config[selector_key]
    kwargs = config.get(name) or {}                                                                   
    return create_fn(name, **kwargs)
                                    

def build_preprocessor(config: dict):                                                                 
    """Construct a preprocessor based on the active selector.

    Returns an object with a .process(wav_path) -> np.ndarray method.
                                                                                                    
    Audio params for log_mel come from the active embedder's config block,
    because the spectrogram shape must match what the embedder was trained                            
    with. The twfr and identity preprocessors don't need that link — twfr                             
    reads its constants from config.py (matching the team's gmm/features.py),                         
    and identity is for pre-extracted data.                                                           
    """                                                                                               
    name = config["preprocessor"]                                                                     
                                                                                                    
    if name == "log_mel":
        from preprocessing.loader import load_audio, split_into_chunks                                
        from preprocessing.compute_mels import log_mel                                                
                                                                                                    
        embedder_name = config["frozen_embedder"]                                                     
        embedder_block = config.get(embedder_name) or {}                                              
                                                                                                    
        if "sample_rate" not in embedder_block or "frame_seconds" not in embedder_block:              
            raise ValueError(                                                                         
                f"log_mel preprocessor needs sample_rate and frame_seconds in "                       
                f"the active embedder's config block ('{embedder_name}'), "                           
                f"but they're missing. The log_mel preprocessor only makes "                          
                f"sense paired with an embedder that declares these params."                          
            )                                                                                         
                                                                                                    
        class LogMelPreprocessor:
            def __init__(self, audio_config: dict):
                self.sample_rate = audio_config["sample_rate"]                                        
                self.frame_seconds = audio_config["frame_seconds"]
                                                                                                    
            def process(self, wav_path: str) -> np.ndarray:
                waveform, sr = load_audio(
                    wav_path, self.sample_rate, mono=True,                                            
                )
                chunks = split_into_chunks(                                                           
                    waveform, sr, self.frame_seconds,                                                 
                )
                return np.stack([log_mel(chunk) for chunk in chunks])                                 
                
        return LogMelPreprocessor(embedder_block)                                                     

    elif name == "twfr":                                                                              
        from gmm.features import load_log_mel

        class TWFRPreprocessor:                                                                       
            def process(self, wav_path: str) -> np.ndarray:
                return load_log_mel(wav_path)                                                         
                
        return TWFRPreprocessor()                                                                     

    elif name == "identity":                                                                          
        class IdentityPreprocessor:
            def process(self, wav_path: str) -> np.ndarray:
                raise NotImplementedError(
                    "Identity preprocessor requires pre-extracted data"                               
                )
                                                                                                    
        return IdentityPreprocessor()

    else:
        raise ValueError(f"Unknown preprocessor: '{name}'")

                                                                                                    
def build_shared_components(config: dict):
    """Construct components shared across all nodes.                                                  
                
    Returns:
        (preprocessor, frozen_embedder, topology, merge)                                              
    """
    preprocessor = build_preprocessor(config)                                                         
                
    frozen_embedder = create_from_config(                                                             
        create_embedder, config, "frozen_embedder",
    )                                                                                                 
                
    topology = create_from_config(create_topology, config, "topology")
    merge = create_from_config(create_merge, config, "merge")
                                                                                                    
    return preprocessor, frozen_embedder, topology, merge
                                                                                                    
                
def build_nodes(
    config: dict,
    preprocessor,
    frozen_embedder,
    topology,
    merge,
    timelines_by_type: dict,                                                                          
) -> dict[str, list[Node]]:
    """Construct all 16 nodes.                                                                        
                                                                                                    
    Shared components are passed in. Each node gets a fresh separator
    instance that owns its own state. The separator's input_dim is                                    
    auto-wired from the active embedder's output dimension.                                           
    """                                                                                               
    separator_name = config["separator"]                                                              
    separator_kwargs = (config.get(separator_name) or {}).copy()                                      
                                                                                                    
    if hasattr(frozen_embedder, "embedding_dim"):
        separator_kwargs["input_dim"] = frozen_embedder.embedding_dim                                 
                                                                                                    
    manual_reset = config.get("simulation", {}).get("manual_reset", False)
                                                                                                    
    nodes_by_type: dict[str, list[Node]] = {}                                                         

    for machine_type, timelines in timelines_by_type.items():                                         
        nodes = []
        for timeline in timelines:
            separator = create_separator(separator_name, **separator_kwargs)
            nodes.append(Node(                                                                        
                node_id=timeline.node_id,
                machine_type=timeline.machine_type,                                                   
                machine_id=timeline.machine_id,
                preprocessor=preprocessor,                                                            
                frozen_embedder=frozen_embedder,
                separator=separator,                                                                  
                topology=topology,
                merge=merge,
                manual_reset=manual_reset,
            ))                                                                                        
        nodes_by_type[machine_type] = nodes
                                                                                                    
    return nodes_by_type