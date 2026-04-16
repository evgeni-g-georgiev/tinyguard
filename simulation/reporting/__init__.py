"""Reporting subpackage — save results, plots, and latent visualisations.                             
                                                                                                    
External callers import from this package directly:
    from simulation.reporting import make_run_dir, save_results, save_plots, save_latent_plots        
                
Internal module boundaries are implementation details.           
"""

from simulation.reporting.results import make_run_dir, save_results 
from simulation.reporting.timeline_plots import save_plots
from simulation.reporting.latent_plots import save_latent_plots 

