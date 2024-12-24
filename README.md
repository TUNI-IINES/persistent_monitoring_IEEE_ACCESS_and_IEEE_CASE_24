# nebolab_simulator
This repository contains simulations used in the [IEEE ACCESS](https://ieeexplore-ieee-org.libproxy.tuni.fi/abstract/document/10379586) and [IEEE CASE 2024](https://ieeexplore-ieee-org.libproxy.tuni.fi/abstract/document/10711560/).

You can read and cite our works:
[1]. Atman, M. W. S., Fikri, M. R., Priandana, K., & Gusrialdi, A. (2024). A Two-Layer Control Framework for Persistent Monitoring of a Large Area With a Robotic Sensor Network. IEEE Access.
[2]. Fikri, M. R., Atman, M. W. S., Nikulin, Y., & Gusrialdi, A. (2024, August). Efficient Multi-Robot Task Allocation with Nonsmooth Objective Functions for Persistent Monitoring in Large Dispersed Areas. In 2024 IEEE 20th International Conference on Automation Science and Engineering (CASE) (pp. 573-578). IEEE

## How to Use?
1. Please ensure the dependencies are installed, such as Numpy, gurobipy, cvxopt, cvxpy, and sklearn.
2. Go to Agrijournal_scenario.py to select the parameters
3. Go to Agrijournal_TaskAllocation.py to double-check the algorithm (MACO used in [1], QP-based used in [2])
4. Run the sim2D_main.py
