import numpy as np
import time
from control_lib.CentVoronoi import CentVoronoiTVDensity
from control_lib.cbf_single_integrator import cbf_si
from scenarios_SI.AgriJournal_TaskAllocation import GenerateSquareField, TaskAllocation, EvaluateAllocation


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from simulator.dynamics import SingleIntegrator
from simulator.plot_2D_pointSI import draw2DPointSI
from simulator.plot_2D_voronoi import DrawVoronoi
from simulator.data_logger import dataLogger
from simulator.timeProfiling import timeProfiling as tp

import pickle # to store and restore task allocation data

# MAIN COMPUTATION
#------------------------------------------------------------------------------
class SceneSetup(): 
    # List of scenario mode
    CLUSTERING_MODE = 0 # clustering broken sensors with K-means
    # CLUSTERING_MODE = 1 # clustering broken sensors with K-medians
    # CLUSTERING_MODE = 2 # clustering broken sensors with Fuzzy-c-means
    # CLUSTERING_MODE = 3 # clustering broken sensors with DPGMM

    GENERATE_SENSOR_MODE = 0 # generate a uniform broken sensor
    # GENERATE_SENSOR_MODE = 1 # generate with a clustered broken sensor
    
    # BOUNDINGBOX_MODE = 0 # generate square bounding box of clustered broken sensor
    BOUNDINGBOX_MODE = 1 # generate convex polygon bounding box of clustered broken sensor
    
    # ALLOCATION_MODE = 0 # allocating with using M-ACO
    ALLOCATION_MODE = 1 # allocating with using QP
    # ALLOCATION_MODE = 2 # allocating with using NLP
    # ALLOCATION_MODE = 3 # allocating with using LP
    # ALLOCATION_MODE = 4 # allocating with using MILP MOO
    # ALLOCATION_MODE = 5 # allocating with using Tchebycheff MOO

    # TOGGLE FOR NEW FEATURE UNDER TESTING. Under tested at the moment.
    IS_COMPUTE_MASS_WITHIN_SENSING_ONLY = True # True (default prev): only compute within sensing region, False: compute whole ROI
    IS_CBF_CONSTRAINT_SCALED = True # True: scaling the constraints to avoid STATUS:unknown, False (default prev): original, without any scaling

    # General variable needed to run the controller
    # Can be adjusted later by set new value on the class variable
    # robot_num = 6
    # init_pos = np.array([[11., 12., 0.], [6., 8., 0.], [6., 10., 0.], [14., 7., 0.], [10., 14., 0.], [17., 3., 0.]])
    robot_num = 4
    # init_pos = np.array([[11., 12., 0.], [6., 8., 0.], [6., 10., 0.], [14., 7., 0.], [10., 14., 0.], [17., 3., 0.],
    #                      [1., 1., 0.], [15., 5., 0.], [10., 18., 0.], [7., 14., 0.]])
    # init_pos = np.array([[11., 12., 0.], [6., 8., 0.], [6., 10., 0.], [14., 7., 0.]])
    # Scale down 
    init_pos = np.array([[10, 7, 0.], [15, 15, 0.], [7, 5, 0.],[14, 10, 0.]])
    
    # Uncomment two lines below to enable the random position based on probability function monte carlo
    # init_pos = np.random.uniform(low=0, high=20, size=(6, 2))
    # init_pos = np.c_[init_pos, np.zeros((6,1))]
    
    # sr_area = [18., 15., 10., 9., 14., 13.]  #6 Agents # scenario 1
    # sr_area = [14., 14., 14., 14., 14., 14.]  #6 Agents # scenario 2
    # sr_area = [30., 36., 40., 34., 42., 46.] #6 Agents # scenario 3
    # sr_area = [1.2, 1., 0.8, 1.5, 0.5, 1.3] #6 Agents
    # sr_area = [15., 16., 17., 18., 19., 20., 21., 22., 23., 24.] # 10 agents for moo 
    sr_area = [30, 32, 34, 36] # 4 agents for moo 
    # sr_area = np.array(random.sample(range(35, 60 ), 6))
    sr_list = [np.sqrt(a/np.pi) for a in sr_area]  #get the sensing radius

    Pgain = 0.8 # for go-to-goal
    # speed_limit = 0. # 0 is no speed limit
    speed_limit = 1. # 0 is no speed limit

    # roi = np.array([0, 0, 20, 20]) # [x_min, y_min, x_max, y_max]
    # Scale down
    roi = np.array([-2., -2., 2., 2.]) # [x_min, y_min, x_max, y_max]
    roi_vertices = np.array([
                [roi[0]-0.1, roi[1]], 
                [roi[2]+0.1, roi[1]], 
                [roi[2], roi[3]], 
                [roi[0], roi[3]],
                [roi[0]-0.1, roi[1]]
            ])

    # PARAMETER TO ADJUST
    # changing the cluster will automatically change the place the data will be stored
    # for more detail, check def_folder. Note that the folder need to be created first.
    # Otherwise it will return with an error (no folder found).
    n_cluster = 2 # Set to 6 only for dpgmm
    # at the moment, below is DEFAULTED with PARAMETER FOR AGRICONTROL SCENARIO
    scene_name = 'agricontrol_'
    broken_sensor_ratio = 0.2
    seed = 11
    calculate_allocation = True # set True it the scenario is generated for the first time 
    # to generate another scenarios, change the values above 

    # TODO: tidy up later
    # grid_size = 1
    # scale down
    grid_size = 0.2
    sensing_rate = 2.5
    # decay_rate = 0.01
    # decay_rate = 0.05
    decay_rate = 0.2 

    # THIS IS EXPERIMENTAL VALUE AT THE MOMENT
    # For more detail on what the value below means, 
    # check nebolab_simulator/__detached_script/AgriJournal_design_density.py
    # gauss_cov_divisor = 4.
    # gaussian_sigma = [[grid_size/gauss_cov_divisor, 0], [0, grid_size/gauss_cov_divisor]] #
    # gaussian_ratio = 0.5
    # gauss_floor_offset = 0.1
    
    # TODO: Cari value yang pas untuk scaling down exp 
    # FINAL Value
    gauss_cov_divisor = 4.
    gaussian_sigma = [[grid_size/gauss_cov_divisor, 0], [0, grid_size/gauss_cov_divisor]] #
    gaussian_ratio = 0.433
    # gauss_floor_offset = 0.3
    gauss_floor_offset = 0.3

    # DEFAULT PARAMETER FOR SAVING RESULTS
    def_folder = 'animation_result/AgriJournalMulti_scenario/' + scene_name + 'k' + str(n_cluster) + '/'
    # Location to save task allocation results
    pkl_taskalloc_name = def_folder + 'ACO_allocation.pkl'
    fplot_clusters = r''+def_folder+'ACO_clusters.pdf'
    fplot_allocate = r''+def_folder+'ACO_allocate.pdf'
    fplot_clustersAllocate = r''+def_folder+'ACO_clustersAllocate.pdf'

    # ds_dyn = 0.15
    ds_dyn = 1.5 # set the safety distance
    gamma_avoidance = 10

    @staticmethod
    def reset_param():
        SceneSetup.pkl_taskalloc_name = SceneSetup.def_folder + 'ACO_allocation.pkl'
        SceneSetup.fplot_clusters = r''+SceneSetup.def_folder+'ACO_clusters.pdf'
        SceneSetup.fplot_allocate = r''+SceneSetup.def_folder+'ACO_allocate.pdf'
        SceneSetup.fplot_clustersAllocate = r''+SceneSetup.def_folder+'ACO_clustersAllocate.pdf'


# General class for computing the controller input
class Controller():
    def __init__(self): # INITIALIZE CONTROLLER
        #################################################
        # TASK ALLOCATION
        if SceneSetup.calculate_allocation:
            x_width = SceneSetup.roi[2] - SceneSetup.roi[0]
            y_width = SceneSetup.roi[3] - SceneSetup.roi[1]

            # Generate field with broken sensors and divide it into k clusters
            field = GenerateSquareField(x_width, y_width, SceneSetup.roi[0], SceneSetup.roi[1], SceneSetup.grid_size) 
            if SceneSetup.GENERATE_SENSOR_MODE == 1:
                p_dens = np.array([ [2., 2., 4], 
                                   [16., 2.5, 3],
                    [2., 17., 3], [15., 17., 4]])                
                field.register_spread_broken_sensor( p_dens )
                print('Generate a clustered broken sensor, MODE: ' + str(SceneSetup.GENERATE_SENSOR_MODE))
                # print(p_dens)
            else:
                print('Generate a uniformed spread of broken sensor, MODE: ' + str(SceneSetup.GENERATE_SENSOR_MODE))
            broken_sensor_pos = field.generate_random_broken_sensor(ratio=SceneSetup.broken_sensor_ratio, seed=SceneSetup.seed)
            
            # Divide the broken sensor position into n number of cluster
            if SceneSetup.CLUSTERING_MODE == 1:
                dict_centroid, dict_dividedPoints = field.kmedians( broken_sensor_pos, SceneSetup.n_cluster)
                print('Clustering ROI based on K-Medians with k:' + str(SceneSetup.n_cluster) \
                    + ', MODE: ' + str(SceneSetup.CLUSTERING_MODE))
            elif SceneSetup.CLUSTERING_MODE == 2:
                dict_centroid, dict_dividedPoints = field.fuzzy_cmeans( broken_sensor_pos, SceneSetup.n_cluster)
                print('Clustering ROI based on Fuzzy-c-means with k:' + str(SceneSetup.n_cluster) \
                    + ', MODE: ' + str(SceneSetup.CLUSTERING_MODE))
            elif SceneSetup.CLUSTERING_MODE == 3:
                dict_centroid, dict_dividedPoints, SceneSetup.n_cluster = field.dpgmm( broken_sensor_pos, SceneSetup.robot_num)
                print('Clustering ROI based on Dirichlet Process GMM with k:' + str(SceneSetup.n_cluster) \
                    + ', MODE: ' + str(SceneSetup.CLUSTERING_MODE))    
            else:
                dict_centroid, dict_dividedPoints = field.kMeans_divide_into_N( broken_sensor_pos, SceneSetup.n_cluster)
                print('Clustering ROI based on K-Means with k:' + str(SceneSetup.n_cluster) \
                    + ', MODE: ' + str(SceneSetup.CLUSTERING_MODE))

            # Create region of interests (ROIs) based on the divided broken sensors: 
            if SceneSetup.BOUNDINGBOX_MODE == 1: # Polygon ROIs versio
                self.dict_boundingBox, dict_task_area = field.generate_convexpoly_for_divided_points( 
                    dict_dividedPoints, SceneSetup.n_cluster, 0.8*SceneSetup.grid_size ) # The offset are larger here than in rectangle case
                print("Area of ROI:", dict_task_area)
                print("Size of Sensing Area:", SceneSetup.sr_area)
                print('Generate convex polygon ROIs of broken sensors, MODE: ' + str(SceneSetup.BOUNDINGBOX_MODE))
            else: # Rectangle ROIs version
                self.dict_boundingBox, dict_task_area = field.generate_boundarybox_for_divided_points( 
                    dict_dividedPoints, SceneSetup.n_cluster, 0.5*SceneSetup.grid_size ) # ROIs are rectangle                
                print('Generate rectangle ROIs of broken sensors, MODE: ' + str(SceneSetup.BOUNDINGBOX_MODE))

            # Allocate drone into the divided regions
            if SceneSetup.ALLOCATION_MODE == 1:
                TAllocate = TaskAllocation(SceneSetup.init_pos, SceneSetup.sr_area, dict_centroid, dict_task_area)
                self.chosenAgentIdx = TAllocate.qp_allocation()
            elif SceneSetup.ALLOCATION_MODE == 2:
                TAllocate = TaskAllocation(SceneSetup.init_pos, SceneSetup.sr_area, dict_centroid, dict_task_area)
                self.chosenAgentIdx = TAllocate.nlp_allocation()
            elif SceneSetup.ALLOCATION_MODE == 3:
                TAllocate = TaskAllocation(SceneSetup.init_pos, SceneSetup.sr_area, dict_centroid, dict_task_area)
                self.chosenAgentIdx = TAllocate.lp_allocation()
            elif SceneSetup.ALLOCATION_MODE == 4:
                TAllocate = TaskAllocation(SceneSetup.init_pos, SceneSetup.sr_area, dict_centroid, dict_task_area)
                self.chosenAgentIdx = TAllocate.multiobjective_allocation()
            elif SceneSetup.ALLOCATION_MODE == 5:
                TAllocate = TaskAllocation(SceneSetup.init_pos, SceneSetup.sr_area, dict_centroid, dict_task_area)
                self.chosenAgentIdx = TAllocate.tchebycheff_allocation
            else: 
                TAllocate = TaskAllocation(SceneSetup.init_pos, SceneSetup.sr_area, dict_centroid, dict_task_area)
                self.chosenAgentIdx = TAllocate.allocate_MACO( )

            # self.chosenAgentIdx = TAllocate.allocate_greedy( )
            # TODO:
            # if SceneSetup.ALLOCATION_MODE == 1: # allocating with using integer programming
            #     self.chosenAgentIdx = TAllocate.integer_programming_allocation( )
            #     print('Allocate task based on integer programming, MODE: ' + str(SceneSetup.ALLOCATION_MODE))
            # else: # allocating with using M-ACO
            #     self.chosenAgentIdx = TAllocate.allocate_MACO( )
            #     print('Allocate task based on M-ACO, MODE: ' + str(SceneSetup.ALLOCATION_MODE))

            # Show the allocation information
            eval = EvaluateAllocation(
                broken_sensor_pos, dict_dividedPoints, self.dict_boundingBox, dict_centroid, dict_task_area,
                SceneSetup.init_pos, SceneSetup.sr_area, self.chosenAgentIdx )
            # print and plots
            eval.print_allocation_info()
            eval.print_agents_task_dist()
            eval.plot_cluster(SceneSetup.fplot_clusters)
            eval.plot_allocation(SceneSetup.fplot_allocate)
            eval.plot_clusterAllocate(SceneSetup.fplot_clustersAllocate)

            # pickle the data for reuse later
            with open(SceneSetup.pkl_taskalloc_name, 'wb') as f:
                pickle.dump((dict_centroid, dict_dividedPoints, self.dict_boundingBox, self.chosenAgentIdx, dict_task_area), f)
                print('Finished storing data of Task Allocation in ' + SceneSetup.pkl_taskalloc_name)

        else: 
            # unpickle already calculated Task allocation
            with open(SceneSetup.pkl_taskalloc_name, 'rb') as f:
                dict_centroid, dict_dividedPoints, self.dict_boundingBox, self.chosenAgentIdx, dict_task_area = pickle.load(f)
                print('Finished loading data of Task Allocation in ' + SceneSetup.pkl_taskalloc_name)

        #################################################
        self.vor = {}
        for i in range(SceneSetup.n_cluster):
            chosenSenRange = [SceneSetup.sr_list[idx] for idx in self.chosenAgentIdx[i]]
            # Initialize voronoi in each region
            self.vor[i] = CentVoronoiTVDensity( SceneSetup.init_pos[self.chosenAgentIdx[i], 0:2], \
                self.dict_boundingBox[i], sensingRadius=chosenSenRange, grid_size=0.8*SceneSetup.grid_size)
            
            # Assign density function from broken sensors in each region
            self.vor[i].set_points_of_importance(dict_dividedPoints[i], SceneSetup.grid_size)
            # self.vor[i].set_density_from_points(dict_dividedPoints[i], sigma=[[0.5*SceneSetup.grid_size, 0], [0, 0.5*SceneSetup.grid_size]], mode='max')
            self.vor[i].set_custom_density_with_floor(dict_dividedPoints[i], 
                SceneSetup.gaussian_sigma, SceneSetup.gaussian_ratio, SceneSetup.gauss_floor_offset)
            
            self.vor[i].set_update_density_rate(SceneSetup.sensing_rate, SceneSetup.decay_rate)
            cent = np.array([dict_centroid[i]])
            self.vor[i].set_centroid( cent ) # for direction before entering the designated roi


        self.cbf = [cbf_si(scale_constraint=SceneSetup.IS_CBF_CONSTRAINT_SCALED) for _ in range(SceneSetup.robot_num)] # For collision avoidance

        # exit()

    def compute_control(self, feedback, computed_control): # INITIALIZE CONTROLLER
        # get all robot position
        pos = feedback.get_all_robot_pos()

        # Reset monitor properties to log data
        computed_control.reset_monitor()

        # Do computation with Voronoi
        vorNomControl = np.zeros((SceneSetup.robot_num, 2))
        centroid = np.zeros((SceneSetup.robot_num, 2))
        for i in range(SceneSetup.n_cluster):
            self.vor[i].update_voronoi_data(pos[self.chosenAgentIdx[i],0:2], dt = feedback.dt)
            reg_centroid, reg_vorNomControl = \
                self.vor[i].compute_nominal_control(use_sensing_rad_tocompute=SceneSetup.IS_COMPUTE_MASS_WITHIN_SENSING_ONLY)

            vorNomControl[self.chosenAgentIdx[i]] = reg_vorNomControl
            centroid[self.chosenAgentIdx[i]] = reg_centroid

            # store the TOTAL coverage metric for each region
            cov_total_ratio, cov_total_current, cov_total_default = self.vor[i].get_total_coverage_ratio()
            computed_control.save_monitored_info("cov_total_ratio_"+str(i), cov_total_ratio)
            computed_control.save_monitored_info("cov_total_current_"+str(i), cov_total_current)
            computed_control.save_monitored_info("cov_total_default_"+str(i), cov_total_default)
            # store the HOLES coverage metric for each region
            cov_hole_ratio, cov_hole_current, cov_hole_default = self.vor[i].get_covhole_coverage_ratio()
            computed_control.save_monitored_info("cov_hole_ratio_"+str(i), cov_hole_ratio)
            computed_control.save_monitored_info("cov_hole_current_"+str(i), cov_hole_current)
            computed_control.save_monitored_info("cov_hole_default_"+str(i), cov_hole_default)


        for i in range(SceneSetup.robot_num):
            # Implementation of Control
            # ------------------------------------------------
            # Calculate nominal controller
            u_nom = np.array([vorNomControl[i,0], vorNomControl[i,1], 0])

            # Construct CBF setup
            self.cbf[i].reset_cbf()
            # Avoid collision to other robots
            current_q = feedback.get_robot_i_pos(i)
            for j in range(SceneSetup.robot_num):
                if i != j:
                    j_q = feedback.get_robot_i_pos(j)
                    h = self.cbf[i].add_avoid_static_circle(current_q, j_q, SceneSetup.ds_dyn, 
                            gamma=SceneSetup.gamma_avoidance, power=1)
                    # store h value
                    computed_control.save_monitored_info( "h_cobs_"+str(i)+"_"+str(j), h )

            if SceneSetup.speed_limit > 0.:
                # # set speed limit on Nominal
                # norm = np.hypot(u_nom[0], u_nom[1])
                # if norm > SceneSetup.speed_limit: u_nom = SceneSetup.speed_limit* u_nom / norm # max 
                # set speed limit on constraint
                self.cbf[i].add_velocity_bound(SceneSetup.speed_limit)

            # Ensure safety
            u = self.cbf[i].compute_safe_controller(u_nom)
            # u = u_nom # Directly use nominal input
            # print(i,u)

            # Store command
            # ------------------------------------------------
            computed_control.set_i_vel_xy(i, u[:2])
            # store information to be monitored/plot
            computed_control.save_monitored_info( "u_nom_x_"+str(i), u_nom[0] )
            computed_control.save_monitored_info( "u_nom_y_"+str(i), u_nom[1] )
            computed_control.save_monitored_info( "u_x_"+str(i), u[0] )
            computed_control.save_monitored_info( "u_y_"+str(i), u[1] )
            computed_control.save_monitored_info( "pos_x_"+str(i), pos[i,0] )
            computed_control.save_monitored_info( "pos_y_"+str(i), pos[i,1] )
            # store the centroid location
            computed_control.save_monitored_info( "vorcent_x_"+str(i), centroid[i,0] )
            computed_control.save_monitored_info( "vorcent_y_"+str(i), centroid[i,1] )

        # Set event-triggered scenario here
        # TODO: Major edit here such as put info about coverage ratio 
        # Step 1: Get the data of current density value of broken sensors
        # Step 2: Get the covhole ratio if lower than 50%, change allocation
        # Step 3: Set a new allocation, change the clustering methods
        # Step 4: Waiting to run and do covering again
        for i in range(SceneSetup.n_cluster):
            datapoints, densval = self.vor[i].extract_mesh_density_data()
            cov_gridPoints, cov_gridDensVal = self.vor[i].extract_covhole_density_data()
            points, points_densVal = self.vor[i].extract_specified_density_data()
            # print(densval)
            # print("Points: ", points) # Data not change: Broken sensor point
            # print("Points val: ", points_densVal)
            # print("Ratio: ", cov_hole_ratio)
            # print("Current Ratio: ", cov_hole_current)
            # print("Default Ratio: ", cov_hole_default)
            # print("Robot pos: ", pos)
            # print("Datapoints:", datapoints[0])
            if cov_hole_ratio <= 0.7:
                # print("Low")
                # self.vor[i].set_density_from_points(datapoints)
                self.vor[i].update_voronoi_data(pos[self.chosenAgentIdx[i],0:2], dt = feedback.dt)
                self.vor[i]._update_bounded_voronoi()

        # Time-Triggered Scenario
        # For every exact 10 seconds, it will change the task allocation.
        # TODO: Figure out how to convert the time similar to time in Sim2D_main
        # for i in range(sim_iter):
        #     current_time = i*SimSetup.Ts
        #     if current_time % 10 == 0: # Every current_time = 2 print hello. This spends 0.02
        #         print("Hello")
                
    
        # Below is a temporary way to pass data for the drawing
        # TODO: think of a tidier way for this
        computed_control.store_voronoi_object(self.vor)
        computed_control.store_taskalloc_data(self.dict_boundingBox, self.chosenAgentIdx)

#-----------------------------------------
# CLASS FOR CONTROLLER'S INPUT AND OUTPUT
#-----------------------------------------
class ControlOutput():
    # Encapsulate the control command to be passed 
    # from controller into sim/experiment
    def __init__(self):
        # Initialize the formation array
        self.__all_velocity_input_xyz = np.zeros([SceneSetup.robot_num, 3])
    
    def get_all_vel_xy(self): return self.__all_velocity_input_xyz[:,:2]

    def get_i_vel_xy(self, ID): return self.__all_velocity_input_xyz[ID,:2]
    def set_i_vel_xy(self, ID, input_xy):
        self.__all_velocity_input_xyz[ID,:2] = input_xy

    # Special case to extract the object within Controller
    def store_voronoi_object(self, obj): self.__vor_obj = obj 
    def get_voronoi_object(self): return self.__vor_obj

    def store_taskalloc_data(self, bbox, agent_idx): 
        self.__task_bbox, self.__task_agent_idx = bbox, agent_idx 
    def get_taskalloc_data(self): return self.__task_bbox, self.__task_agent_idx
    

    # Allow the options to monitor state / variables over time
    def reset_monitor(self): self.__monitored_signal = {}
    def save_monitored_info(self, label, value): 
        # NOTE: by default name the label with the index being the last
        # example p_x_0, p_y_0, h_form_1_2, etc.
        self.__monitored_signal[label] = value
    # Allow retrieval from sim or experiment
    def get_all_monitored_info(self): return self.__monitored_signal


class FeedbackInformation():
    # Encapsulate the feedback information to be passed 
    # from sim/experiment into controller
    def __init__(self):
        # Set the value based on initial values
        self.set_feedback(SceneSetup.init_pos, 0.)

    # To be assigned from the SIM or EXP side of computation
    def set_feedback(self, all_robots_pos, dt):
        self.dt = dt # to allow discrete computation within internal controller
        # update all robots position and theta
        self.__all_robot_pos = all_robots_pos.copy()

    # To allow access from the controller computation
    def get_robot_i_pos(self, i):   return self.__all_robot_pos[i,:]
    # get all robots information
    def get_all_robot_pos(self):   return self.__all_robot_pos


# ONLY USED IN SIMULATION
#-----------------------------------------------------------------------
class SimSetup():

    Ts = 0.02 # in second. Determine Visualization and dynamic update speed
    tmax = 30. # simulation duration in seconds (only works when save_animate = True)
    save_animate = True # True: saving but not showing, False: showing animation but not real time
    save_data = True # log data using pickle
    plot_saved_data = True # plot saved data (save_data should be True)
    clean_version = True # True: only show density function on specified point

    sim_defname = SceneSetup.def_folder + 'sim'
    sim_fname_output = r''+sim_defname+'.gif'
    trajectory_trail_lenTime = 15 #tmax # Show all trajectory
    sim_fdata_log = sim_defname + '_data.pkl'

    timeseries_window = tmax #5 # in seconds, for the time series data

    field_x = [SceneSetup.roi[0]-1., SceneSetup.roi[2]+1.]
    field_y = [SceneSetup.roi[1]-1., SceneSetup.roi[3]+1.]


    # snap_figure = [15, 30, 45] # in second
    snap_figure = [] # in second

    @staticmethod
    def reset_param():
        SimSetup.sim_defname = SceneSetup.def_folder + 'sim'
        SimSetup.sim_fname_output = r''+SimSetup.sim_defname+'.gif'
        SimSetup.trajectory_trail_lenTime = SimSetup.tmax # Show all trajectory
        SimSetup.sim_fdata_log = SimSetup.sim_defname + '_data.pkl'


# General class for drawing the plots in simulation
class SimulationCanvas():
    def __init__(self):
        self.__sim_ctr = 0
        self.__max_ctr = round(SimSetup.tmax / SimSetup.Ts)
        self.__cur_time = 0.

        # Initiate the robot
        self.__robot_dyn = [None]*SceneSetup.robot_num
        for i in range(SceneSetup.robot_num):
            self.__robot_dyn[i] = SingleIntegrator(SimSetup.Ts, SceneSetup.init_pos[i])

        # Initiate data_logger
        self.log = dataLogger( self.__max_ctr )
        # Initiate the plotting
        self.__initiate_plot()

        # flag to check if simulation is still running
        self.is_running = True

        # Functionality to save figure at certain time
        self.snap_idx, self.snap_time = -1, -1
        if len(SimSetup.snap_figure) > 0:
            self.snap_idx = 0
            self.snap_time = SimSetup.snap_figure[self.snap_idx]


    def update_simulation(self, control_input, feedback):
        if ( self.__sim_ctr < self.__max_ctr ):
            # Store data to log
            self.log.store_dictionary( control_input.get_all_monitored_info() )
            self.log.time_stamp( self.__cur_time )
            # Update plot
            self.__update_plot( feedback, control_input )
    
            self.__sim_ctr += 1
            self.__cur_time = self.__sim_ctr * SimSetup.Ts
            # Set array to be filled
            all_robots_pos = np.zeros( SceneSetup.init_pos.shape )
            # IMPORTANT: advance the robot's dynamic, and update feedback information
            for i in range(SceneSetup.robot_num):
                self.__robot_dyn[i].set_input(control_input.get_i_vel_xy(i), "u")
                state = self.__robot_dyn[i].step_dynamics() 
                all_robots_pos[i,:2] = state['q'][:2]
    
            # UPDATE FEEDBACK for the controller
            feedback.set_feedback(all_robots_pos, SimSetup.Ts)

            if self.__cur_time == self.snap_time:
                f_out = SimSetup.sim_defname+str(self.__cur_time)+'.pdf'
                plt.savefig(f_out, bbox_inches="tight")
                print('Saving a snap at t:'+str(self.__cur_time)+' into '+f_out)

                # advance snap time
                self.snap_idx += 1
                if len(SimSetup.snap_figure) - self.snap_idx >= 1:
                    self.snap_time = SimSetup.snap_figure[self.snap_idx]
                else:
                    self.snap_idx, self.snap_time = -1, -1


        else: # No further update
            if self.is_running:
                if SimSetup.save_data: 
                    self.log.save_to_pkl( SimSetup.sim_fdata_log )
                    if SimSetup.plot_saved_data: 
                        from scenarios_SI.AgriJournal_pickleplot import scenario_pkl_plot
                        scenario_pkl_plot()
                print( f"Stopping the simulation, tmax reached: {self.__cur_time:.2f} s" )
                # if not SimSetup.save_animate: exit() # force exit
                self.is_running = False 
            # else: # Do nothing


    # PROCEDURES RELATED TO PLOTTING - depending on the scenarios
    #---------------------------------------------------------------------------------
    def __initiate_plot(self):
        # Initiate the plotting
        # For now plot 2D with 2x2 grid space, to allow additional plot later on
        rowNum, colNum = 2, 2
        self.fig = plt.figure(figsize=(4*colNum, 3*rowNum), dpi= 100)
        gs = GridSpec( rowNum, colNum, figure=self.fig)

        # MAIN 2D PLOT FOR UNICYCLE ROBOTS
        # ------------------------------------------------------------------------------------
        ax_2D = self.fig.add_subplot(gs[0:2,0:2]) # Always on
        # Only show past several seconds trajectory
        trajTail_datanum = int(SimSetup.trajectory_trail_lenTime/SimSetup.Ts) 

        self.__drawn_2D = draw2DPointSI( ax_2D, SceneSetup.init_pos,
            field_x = SimSetup.field_x, field_y = SimSetup.field_y, pos_trail_nums=trajTail_datanum )
        
        # Map the ticks for x and y axes
        ax_2D.set_xticks([ 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20 ])
        ax_2D.set_xticklabels([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])  # Original range mapping
        ax_2D.set_yticks([ 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20 ])
        ax_2D.set_yticklabels([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])  # Original range mapping

        # Draw goals and obstacles
        v0_robnum = np.zeros(SceneSetup.robot_num)
        self.__pl_goals, = ax_2D.plot(v0_robnum, v0_robnum, 'r.')

        # Display simulation time
        self.__drawn_time = ax_2D.text(0.78, 0.99, 't = 0 s', color = 'k', fontsize='large', 
            horizontalalignment='left', verticalalignment='top', transform = ax_2D.transAxes)
        # TODO: incorporate logger
        # TODO: plot the h-function, etc

        # Initiate voronoi diagram
        self.__drawn_vor = {i:None for i in range(SceneSetup.n_cluster)}
        self.__ax_2D = ax_2D

        # # ADDITIONAL PLOT
        # # ------------------------------------------------------------------------------------
        # # Plot nominal velocity in x- and y-axis
        # self.__ax_unomx = self.fig.add_subplot(gs[0,2])
        # self.__ax_unomy = self.fig.add_subplot(gs[1,2])
        # self.log.plot_time_series_batch( self.__ax_unomx, 'u_x_' ) 
        # self.log.plot_time_series_batch( self.__ax_unomy, 'u_y_' ) 
        # # Plot position in x- and y-axis
        # self.__ax_pos_x = self.fig.add_subplot(gs[2,0])
        # self.__ax_pos_y = self.fig.add_subplot(gs[2,1])
        # self.log.plot_time_series_batch( self.__ax_pos_x, 'pos_x_' ) 
        # self.log.plot_time_series_batch( self.__ax_pos_y, 'pos_y_' ) 

        plt.tight_layout()


    def __update_plot(self, feedback, control_input):
        # UPDATE 2D Plotting: Formation and Robots
        self.__drawn_2D.update( feedback.get_all_robot_pos() )
        self.__drawn_time.set_text('t = '+f"{self.__cur_time:.1f}"+' s')

        # update each robot's target
        updated_goals = np.zeros((SceneSetup.robot_num, 3))
        for i in range(SceneSetup.robot_num):
            updated_goals[i,0] = self.log.get_lastdata_from_label("vorcent_x_"+str(i))
            updated_goals[i,1] = self.log.get_lastdata_from_label("vorcent_y_"+str(i))
        self.__pl_goals.set_data(updated_goals[:,0], updated_goals[:,1])

        # Plot the voronoi diagram
        vor_obj = control_input.get_voronoi_object()
        bbox, agent_idx = control_input.get_taskalloc_data()
        for i in range(SceneSetup.n_cluster):
            if self.__drawn_vor[i] is None:
                self.__drawn_vor[i] = DrawVoronoi(self.__ax_2D, bbox[i], len(agent_idx[i]), 
                    color_idx = agent_idx[i])
        
            # extract the required data from vor_obj and use for drawing
            # voronoi vertices
            vor_vertices = vor_obj[i].extract_voronoi_vertices()
            self.__drawn_vor[i].plot_voronoi_diagram(vor_vertices)
            # density function
            if not SimSetup.clean_version: # whole field
                datapoints, density_val = vor_obj[i].extract_mesh_density_data()
                self.__drawn_vor[i].plot_density_function(datapoints, density_val)
            else: # only on specified point
                # specified_points, corresponding_density = vor_obj[i].extract_specified_density_data()
                specified_points, corresponding_density = vor_obj[i].extract_covhole_density_data()
                self.__drawn_vor[i].plot_density_specified_points(specified_points, corresponding_density)
            # the sensing area
            sensing_area_data = vor_obj[i].extract_sensing_data()
            self.__drawn_vor[i].plot_sensing_area(sensing_area_data)

        # # get data from Log
        # log_data, max_idx = self.log.get_all_data()
        # # Setup for moving window horizon
        # min_idx = 0
        # if (self.__cur_time > SimSetup.timeseries_window): 
        #     min_idx = max_idx - round(SimSetup.timeseries_window/SimSetup.Ts)

        # # update nominal velocity in x- and y-axis
        # self.log.update_time_series_batch( 'u_x_', data_minmax=(min_idx, max_idx)) 
        # self.log.update_time_series_batch( 'u_y_', data_minmax=(min_idx, max_idx)) 

        # # update position in x- and y-axis
        # self.log.update_time_series_batch( 'pos_x_', data_minmax=(min_idx, max_idx)) 
        # self.log.update_time_series_batch( 'pos_y_', data_minmax=(min_idx, max_idx)) 
