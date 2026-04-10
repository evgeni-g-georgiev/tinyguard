# Simulation Module

  Lockstep simulation of on-device anomalous sound detection across all 16            
  MIMII machines (4 types × 4 IDs). Runs every node through the same time
  sequence in parallel, exactly as real devices would experience it.                  
                  
  ## What it does                                                                     
                  
  For each configured pipeline (CNN+SVDD or TWFR+GMM):                                
                  
  1. Loads all 16 nodes' audio splits                                                 
  2. Calibrates each node's separator on its warmup clips
  3. Steps through the test timeline one clip at a time, scoring every                
     node simultaneously                                                              
  4. Saves results, AUC tables, and plots to a timestamped output directory           
                                                                                      
  ## Setup (one-time)                                                                 
                                                                                      
  ```bash         
  # 1. Activate the project venv
  source .venv/bin/activate                                                           
   
  # 2. Download MIMII data (~30 GB, takes ~1 hour)                                    
  python data/download_mimii.py
                                                                                      
  # 3. Create the simulation splits (warmup / test_normal / test_abnormal             
  #    per node, symlinked from data/mimii to avoid duplication)                      
  python -m simulation.data.split_data                                                
                                                                                      
  Run                                                                                 
                                                                                      
  python -m simulation.run_simulation

  Reads simulation/configs/default.yaml. Switch pipelines by changing                 
  three selector keys at the top of the YAML:
                                                                                      
  preprocessor:    log_mel | twfr
  frozen_embedder: acoustic_encoder | identity                                        
  separator:       svdd | gmm | identity                                              
                                                                                      
  Output                                                                              
                                                                                      
  Each run creates a fresh directory under simulation/outputs/runs/:

  simulation/outputs/runs/2026-04-10_14-23-05/                                        
    config.yaml          (verbatim copy of the YAML used)                             
    results.json         (per-node scores, labels, AUC)                               
    summary.txt          (human-readable AUC table with metadata)                     
    plots/                                                                            
      grid.png           (4×4 grid of all nodes' score timelines)
      fan_id_00.png      (per-node score timeline)                                    
      ...                                                                             
      latent/                                                                         
        grid_per_frame.png  (4×4 grid of t-SNE plots)                                 
        grid_per_clip.png                                                             
        fan_id_00.png       (3-panel: t-SNE + histogram per node)
        ...                                                                           
                  
  config.yaml and results.json are sufficient to fully reproduce the                  
  plots — the simulation never overwrites previous runs.
                                                                                      
  Disabling slow steps
                                                                                      
  Latent t-SNE plots add ~1-2 minutes per run. Disable them while iterating:          
   
  latent_plot:                                                                        
    enabled: false