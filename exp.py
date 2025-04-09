import os
import sys
import matplotlib.pyplot as plt
from ar1 import run_ar1_experiment
from kalman import run_kalman_experiment
from city_a import run_city_experiment

def run_table2():
    """Experiment for Table 2: Unit output analysis on AR1 dataset"""
    results = run_ar1_experiment()
    rmse, max_val, min_val, mean_val, fig = results
    
    # Save figure
    os.makedirs('./figures', exist_ok=True)
    fig.savefig('./figures/ar1_unit.png')
    plt.close()
    
    return rmse, max_val, min_val, mean_val

def run_table3():
    """Experiment for Table 3: Aggregate output analysis on AR1 and Kalman datasets"""
    # Run AR1 experiment
    ar1_rmse, ar1_fig = run_ar1_experiment()
    
    # Run Kalman experiment
    kalman_rmse, kalman_fig = run_kalman_experiment()
    
    # Save figures
    os.makedirs('./figures', exist_ok=True)
    ar1_fig.savefig('./figures/ar1_aggregate.png')
    kalman_fig.savefig('./figures/kalman_capacity.png')
    plt.close('all')
    
    return {'ar1_rmse': ar1_rmse, 'kalman_rmse': kalman_rmse}

def run_table4():
    """Experiment for Table 4: City A dataset analysis"""
    return run_city_experiment()

def run_figure9():
    """Generate and plot results for Figure 9 using Kalman dataset and AR1 dataset"""
    # Get Kalman results and plot
    kalman_rmse, kalman_fig = run_kalman_experiment()
    
    # Get AR1 results and plot
    ar_rmse, ar_fig = run_ar1_experiment()
    
    # Save both figures
    os.makedirs('./figures', exist_ok=True)
    kalman_fig.savefig('./figures/figure9_capacity.png')
    ar_fig.savefig('./figures/ar_capacity.png')
    plt.close('all')
    
    print("Capacity plots have been saved to ./figures/")
    # save rmse in txt
    os.makedirs('./rmse', exist_ok=True)
    with open('./rmse/figure9_rmse.txt', 'w') as f:
        f.write(f"Kalman RMSE: {kalman_rmse:.4f}\n")
        f.write(f"AR RMSE: {ar_rmse:.4f}\n")
    print("RMSEs have been saved to ./rmse/figure9_rmse.txt")

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python exp.py [table2|table3|table4|figure9]")
    #     sys.exit(1)
    
    # experiment = sys.argv[1].lower()
    experiment = "figure9"
    
    if experiment == "figure9":
        run_figure9()
    elif experiment == "table2":
        rmse, max_val, min_val, mean_val = run_table2()
        print(f"Table 2 Results:")
        print(f"RMSE: {rmse:.4f}")
        print(f"Max: {max_val:.4f}")
        print(f"Min: {min_val:.4f}")
        print(f"Mean: {mean_val:.4f}")
        print("Plots have been saved to ./figures/")
    elif experiment == "table3":
        results = run_table3()
        print(f"Table 3 Results:")
        print(f"AR1 RMSE: {results['ar1_rmse']:.4f}")
        print(f"Kalman RMSE: {results['kalman_rmse']:.4f}")
        print("Plots have been saved to ./figures/")
    elif experiment == "table4":
        rmse = run_table4()
        print(f"Table 4 Results:")
        print(f"City A RMSE: {rmse:.4f}")
    else:
        print("Invalid experiment name. Choose from: table2, table3, table4, figure9")
        sys.exit(1)