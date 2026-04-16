# Simulation                                                                                          
                                                                                                      
Lockstep simulation of on-device anomalous sound detection across 16 MIMII                            
machines (4 types × 4 IDs).
                                                                                                      
## Quickstart   

```bash
# 1. Download MIMII data for one SNR (~30 GB per SNR)
python data/download_mimii.py --snr 6dB      # also: 0dB, -6dB                                        
                                                                                                      
# 2. Split into warmup / test_normal / test_abnormal per node                                         
python -m simulation.data.split_data --snr 6dB                                                        
                
# 3. Run the simulation
python -m simulation.run_simulation
                                                                                                      
Step 2 is optional — the simulation auto-splits on first run if the splits                            
dir is missing. Step 1 is not optional; raw MIMII data must be on disk.                               
                                                                                                      
Switching SNR   
                                                                                                      
Edit simulation/configs/default.yaml:                                                                 
snr: 0dB    # 6dB | 0dB | -6dB
                                                                                                      
Or download + split a new SNR and re-run. Each SNR lives in its own dir
(data/mimii/<snr>/, simulation/data/splits/<snr>/), so they coexist.                                  
  
Output                                                                                                
                
Each run creates a timestamped directory under simulation/outputs/runs/:                              
                
simulation/outputs/runs/2026-04-16_10-30-00/
  config.yaml       — verbatim copy of the YAML used
  results.json      — per-node scores, labels, AUC, confusion                                         
  summary.txt       — human-readable AUC table
  plots/                                                                                              
    grid.png        — 4×4 score timeline grid                                                         
    grid_state.png  — 4×4 with state detection brackets (when state_enabled)
    fan_id_00.png   — per-node full-size timeline                                                     
    ...         
    latent/         — t-SNE and score distribution grids (when latent_plot.enabled)                   
                                                                                                      
Module layout
                                                                                                      
simulation/     
  run_simulation.py   — entry point: reads YAML, wires everything, runs lockstep
  builders.py         — constructs components from config (preprocessor, embedder, etc.)              
  metrics.py          — pure metric computation (NodeMetrics, BlockStateMetrics)                      
  formatters.py       — terminal output formatting (print_results, format_step)                       
  lockstep.py         — the lockstep evaluation loop                                                  
  registry.py         — component registry (register/create pattern)                                  
  reporting/          — all disk IO: results serialisation + matplotlib plots                         
  configs/            — YAML experiment configs                                                       
  data/               — data loading, splitting, timeline construction
  inference_models/   — separator and embedder implementations                                        
  node/               — Node class, topology, merge

Disabling slow steps

Latent t-SNE plots 
latent_plot:
  enabled: false      
---

                