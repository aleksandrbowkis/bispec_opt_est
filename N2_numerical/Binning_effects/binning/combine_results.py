""" Code to combine results from multiple jobs (one for each different bin). See Calc_binning for more details. """

import numpy as np

# combine results after all tasks complete
def combine_results():
    series_str = 'series' if is_it_series else 'no_series'
    
    # Initialize lists to store all results
    all_bin_mids = []
    all_N2_equi = []
    all_N2_fold = []
    
    # Load results from each task
    for task_id in range(12):  # Assuming 12 tasks
        eq_data = np.load(f'../outputs/{series_str}_binned_equilateral_task_{task_id}.npy')
        fd_data = np.load(f'../outputs/{series_str}_binned_folded_task_{task_id}.npy')
        
        all_bin_mids.extend(eq_data[0])
        all_N2_equi.extend(eq_data[1])
        all_N2_fold.extend(fd_data[1])
    
    # Save combined results
    np.save(f'../outputs/{series_str}_binned_equilateral_combined.npy', 
            (np.array(all_bin_mids), np.array(all_N2_equi)))
    np.save(f'../outputs/{series_str}_binned_folded_combined.npy', 
            (np.array(all_bin_mids), np.array(all_N2_fold)))
    
if __name__ == '__main__':
    is_it_series = False  # Set to True if calculating the series approximation for N2
    combine_results()
    print('Results combined successfully!')