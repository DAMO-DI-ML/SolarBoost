import os
import sys
import matplotlib.pyplot as plt
from ar1 import run_ar1_experiment
from kalman import run_kalman_experiment
from city_a import run_city_experiment

def run_table2():
    ar_res = run_ar1_experiment(goal = ['rmse_y1','output'])
    
    # save rmse in txt
    os.makedirs('./table', exist_ok=True)
    with open('./table/table2_ar_grid1.txt', 'w') as f:
        f.write(f"RMSE: {ar_res['rmse_y1']:.4f}\n")
        f.write(f"max: {ar_res['output_max']:.4f}\n")
        f.write(f"min: {ar_res['output_min']:.4f}\n")
        f.write(f"mean: {ar_res['output_mean']:.4f}\n")
    print("ar_grid1 results have been saved to ./table/table2_ar_grid1.txt")
    

def run_table3():
    """Experiment for Table 3: Aggregate output analysis on AR1 and Kalman datasets"""
    # Run AR1 experiment
    ar_rmse = run_ar1_experiment(goal = ['rmse_y'])['rmse_y']
    
    # Run Kalman experiment
    kalman_rmse= run_kalman_experiment(goal = ['rmse_y'])['rmse_y']

    # save rmse in txt
    os.makedirs('./table', exist_ok=True)
    with open('./table/table3_rmse.txt', 'w') as f:
        f.write(f"Kalman RMSE: {kalman_rmse:.4f}\n")
        f.write(f"AR RMSE: {ar_rmse:.4f}\n")
    print("RMSEs have been saved to ./table/table3_rmse.txt")

def run_table4():
    """Experiment for Table 4: City A dataset analysis"""
    rmse = run_city_experiment()['rmse_y']
    print(f"Table 4 Results:")
    print(f"City A RMSE: {rmse:.4f}")
    return rmse

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


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python exp.py [table2|table3|table4|figure9]")
    #     sys.exit(1)
    
    # experiment = sys.argv[1].lower()
    experiment = "table4"
    
    if experiment == "figure9":
        run_figure9()
    elif experiment == "table2":
        run_table2()
    elif experiment == "table3":
        run_table3()
    elif experiment == "table4":
        run_table4()
    else:
        print("Invalid experiment name. Choose from: table2, table3, table4, figure9")
        sys.exit(1)