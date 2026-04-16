# Data pipeline 

## Download

```bash
python data/download_mimii.py --snr 6dB   # ~30 GB, one SNR at a time
python data/download_mimii.py --snr 0dB                                                               
python data/download_mimii.py --snr -6dB                                                              
                                                                                                    
Downloads from Zenodo record 3384388. Each SNR variant extracts to                                    
data/mimii/<snr>/{fan,pump,slider,valve}/{id_00,id_02,id_04,id_06}/.
                                                                                                    
First run auto-migrates any legacy flat data/mimii/fan/... layout into                                
data/mimii/6dB/fan/....                                                                               
                                                                                                    
Split           

python -m simulation.data.split_data --snr 6dB

Produces simulation/data/splits/<snr>/ with symlinks into the raw data:
simulation/data/splits/6dB/
fan/id_00/{warmup, test_normal, test_abnormal}/*.wav
...                                                 
surplus_abnormal/fan/id_00/*.wav
                                
- warmup — calibration data (size controlled by simulation.warmup_count)
- test_normal / test_abnormal — balanced counts across all 16 nodes                                   
- surplus_abnormal — leftover anomaly clips not used in test                                          
                                                                                                    
The simulation auto-runs this step if the splits dir is missing.                                      
                                                                                                    
Files                                                                                                 
                
- split_data.py — planning + execution of the warmup/test split                                       
- simulation_loader.py — loads splits into NodeTimeline objects with
configurable shuffle modes (random, block_random, block_fixed)                                        
                
---    