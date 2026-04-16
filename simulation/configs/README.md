```markdown
# Config reference

`default.yaml` is the single source of truth for a simulation run.                                    
Copy it to create experiment variants:
                                                                                                    
```bash         
cp simulation/configs/default.yaml simulation/configs/my_experiment.yaml                              
python -m simulation.run_simulation --config simulation/configs/my_experiment.yaml
                                                                                                    
Key sections
                                                                                                    
Pipeline selection — top-level selectors choose which implementations to use:                         
preprocessor:    twfr          # log_mel | twfr | identity
frozen_embedder: identity      # acoustic_encoder | identity                                          
separator:       gmm           # svdd | gmm | identity      
                                                                                                    
Separator params — each separator has its own named block:
gmm:                                                                                                  
n_components: 2
covariance_type: diag                                                                               
threshold_mode: percentile   # percentile | max_margin | n_sigma
threshold_percentile: 99.0
                                                                                                    
Simulation runtime:
simulation:                                                                                           
warmup_count: 400            # clips used to calibrate each node
shuffle_mode: block_fixed    # random | block_random | block_fixed
block_size: 5                # anomaly clips per block                                              
block_interval: 4            # normal clips between blocks                                          
state_enabled: true          # print block-level state metrics                                      
manual_reset: true           # circuit-breaker mode (engineer resets)                               
                                                                                                    
SNR variant:
snr: 6dB                      # 6dB | 0dB | -6dB                                                      
data:                                           
mimii_root: data/mimii/{snr}                                                                        
splits_dir: simulation/data/splits/{snr}
                                                                                                    
State scoring
                                                                                                    
The simulation tracks two levels of prediction per clip:                                              

1. Clip-level (always on) — score > threshold per clip. Gives AUC, P, R, F1.                          
2. Block-level (when state_enabled: true) — temporal state that tracks "am I
currently in an anomaly state?" Gives block P/R/F1, detection lag, recovery time.                     
                                                                                                    
Two modes controlled by manual_reset:                                                                 
- true — circuit breaker: once the state fires, it holds until the next normal                        
block boundary (simulates engineer arriving to reset). Unflag metric is N/A.                          
- false — auto: state returns to normal when the separator score drops below
threshold. Measures both detection lag and unflag recovery time.                                      
                                                                                                    
Adding a new separator                                                                                
                                                                                                    
1. Create simulation/inference_models/separator_mine.py                                               
2. Subclass BaseOnDeviceSeparator, implement calibrate, score, description,
project, get_shareable_state, merge_state                                                             
3. Decorate with @register_separator("mine")                                                          
4. Import it in simulation/__init__.py                                                                
5. Add a mine: param block in your YAML and set separator: mine                                       
                                                                                                    
To add a custom state scorer, override state(score, **kwargs) and                                     
reset_state() on your separator class. The defaults work for most cases.                              
                                                                                                    
---             