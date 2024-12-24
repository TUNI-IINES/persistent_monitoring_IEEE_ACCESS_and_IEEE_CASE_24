import matplotlib.pyplot as plt
import time
import numpy as np
import random
import gurobipy as gp
import cvxpy as cp

from gurobipy import GRB
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import DBSCAN

class GenerateSquareField():
    def __init__(self, x_width, y_width, x_origin=0., y_origin=0., grid_size=1.):
        self.width = [x_width, y_width]
        self.origin = [x_origin, y_origin]
        self.grid_size = grid_size # Assuming uniform placing of sensors

        p = int( x_width // grid_size )
        l = int( y_width // grid_size )
        self.sensorStatus = np.ones( (p,l), dtype=int)
        self.pdf_spread = None # Uniform distribution

    # Each row should be specified with [center_x, center_y, radius]
    # p_dens = np.array([
    #     [4., 4., 4],
    #     [16., 7., 3],
    #     [10., 17., 5]
    # ])
    def register_spread_broken_sensor(self, p_dens):
        from scipy.stats import multivariate_normal
        
        i,j = np.nonzero(self.sensorStatus)
        sensorLoc = np.transpose(np.array([i, j]))*self.grid_size + self.origin
        
        pdf = np.zeros(len(i))
        for p in p_dens:
            rv = multivariate_normal([p[0], p[1]], np.eye(2)*p[2])
            pdf = np.maximum(pdf, rv.pdf(sensorLoc))

        self.pdf_spread = pdf/np.sum(pdf)


    def generate_random_broken_sensor(self, ratio=0.2, seed=None):
        # Randomly generate 20% of broken sensors
        i,j = np.nonzero(self.sensorStatus)
        if seed is not None: np.random.seed(seed)
        ix = np.random.choice(len(i), int(np.floor(ratio * len(i))), replace=False, p = self.pdf_spread)
        self.sensorStatus[i[ix], j[ix]] = 0

        # Broken sensors' positions
        points = []
        for n in np.unique(self.sensorStatus):
            if n == 0:
                pos = np.argwhere(self.sensorStatus == n)
                pos = pos*self.grid_size + self.origin # shift according to gridsize and origin
                points.append(pos)

        # self.BrokenSensorPos = np.vstack(points)
        return np.vstack(points)

    def generate_random_agents_pos(self, n): # Within Field
        agentsPositions = np.zeros([n,3])
        for i in range (n):
            x = random.randint(self.origin[0], self.origin[0]+self.width[0])
            y = random.randint(self.origin[1], self.origin[1]+self.width[1])
            agentsPositions[i] = np.array( [ x, y, 0 ] )

        return agentsPositions

# --------------------------------------------------------------------------
# This section is focusing on Clustering algorithm for sensor allocation
# --------------------------------------------------------------------------
    @staticmethod
    def dpgmm(points, max_clusters):
        
        dict_centroids = {}
        centroids = {}

        # please choose weight_concentration_prior carefully
        dpgmm = BayesianGaussianMixture(n_components=max_clusters,
                                    max_iter=1000,
                                    weight_concentration_prior=0.1,
                                    init_params='kmeans',
                                    weight_concentration_prior_type='dirichlet_process',
                                    n_init=10)
        
        # Use fit_predict(points) to less the redundancy in the code
        dpgmm_labels = dpgmm.fit_predict(points)

        weights_dpgmm = np.round(dpgmm.weights_,2)

        # Count non-zero values indicate the number of clusters
        count = sum(1 for element in weights_dpgmm if element != 0)

        centroids = dpgmm.means_
        dict_centroids =  centroids[weights_dpgmm > weights_dpgmm.max() / 10]
        # print("Centroids :", dict_centroids)
        # print("Weight", weights_dpgmm)
        
        # Divide list of points & Get the bounding box
        dict_dividedPoints = { i: points[dpgmm_labels == i] for i in range(max_clusters) }
        return dict_centroids, dict_dividedPoints, count        

    
    @staticmethod
    def fuzzy_cmeans(points, k, max_iteration = 100):
        from scipy.spatial.distance import cdist
        m = 3
        # Define the variable of membership
        U = np.random.rand(points.shape[0],k)
        U /= np.sum(U, axis=1)[:,np.newaxis]

        # Declare the k centroids randomly
        def calculate_centroid (points, k, U, m):
            centroids = np.zeros((k, points.shape[1]))
            for i in range (k):
                centroids[i,:] = np.sum((U[:,i] ** m)[:,np.newaxis] * points, axis = 0) / np.sum(U[:,i] ** m)
            return centroids

        # Calculate new membership
        def calculate_membership (points, centroids, k , m):
            U_new = np.zeros((points.shape[0], k))
            for i in range (k):
                U_new[:,i] = np.linalg.norm(points - centroids[i,:], axis=1)

            U_new = 1 / (U_new ** (2/(m-1)) * np.sum((1/U_new) ** (2/(m-1)) , axis = 1 )[:, np.newaxis] )
            return U_new

        # Make the cluster has a better resolution
        for iteration in range (max_iteration):
            centroids = calculate_centroid(points, k, U , m)
            U_new = calculate_membership(points, centroids, k , m) 
      
            if np.linalg.norm (U_new - U) <= 0.00001:
                break
            
            U = U_new

        labels = np.argmax(U_new, axis=1)
        dict_dividedPoints = { i: points[labels == i] for i in range(k) }

        return centroids, dict_dividedPoints
    

    @staticmethod
    def kmedians(points, k, max_iterations=1000):
        from scipy.spatial.distance import cdist

        # Initialize k centroids randomly
        centroids = points[np.random.choice(range(len(points)), k, replace=False)]

        
        for _ in range(max_iterations):
            # Assign each data point to its closest centroid
            distances = cdist(points, centroids, metric='cityblock')
            labels = np.argmin(distances, axis=1)

            # Update the cluster centers using medians
            new_centers = np.array([np.median(points[labels == i], axis=0) for i in range(k)])

            # Check convergence
            if np.array_equal(centroids, new_centers):
                break

            centroids = new_centers

            # prev_labels = labels.copy()
            dict_dividedPoints = { i: points[labels == i] for i in range(k) }
        # Return the centroids and labels
        return centroids, dict_dividedPoints

    @staticmethod
    def kMeans_divide_into_N(points, n):
        from sklearn.cluster import KMeans
        # K-Means clustering to divide the area of broken sensors
        kmeans = KMeans(n, random_state=0).fit(points)
        # kmeans = KMeans(n, init=TaskAllocation.).fit(points)
        labels = kmeans.labels_
        dict_centroid = kmeans.cluster_centers_
        # Divide list of points & Get the bounding box
        dict_dividedPoints = { i: points[labels == i] for i in range(n) }

        return dict_centroid, dict_dividedPoints

# --------------------------------------------------------------------------------------------------
# This part for boundary setting
# --------------------------------------------------------------------------------------------------
    @staticmethod
    def generate_boundarybox_for_divided_points(dict_dividedPoints, n_cluster, cluster_offset = 0.):
        dict_boundingBox = {}
        dict_task_area = np.zeros(n_cluster) #define task area based on ROI
        for i in range(n_cluster):
            bb = [  min(dict_dividedPoints[i][:,0])-cluster_offset, 
                    min(dict_dividedPoints[i][:,1])-cluster_offset, 
                    max(dict_dividedPoints[i][:,0])+cluster_offset, 
                    max(dict_dividedPoints[i][:,1])+cluster_offset ] # xmin, ymin, xmax, ymax
            width, height = (bb[2]-bb[0]), (bb[3]-bb[1])

            dict_boundingBox[i] = np.array( [
                [bb[0], bb[2], bb[2], bb[0], bb[0]],
                [bb[1], bb[1], bb[3], bb[3], bb[1]] ]).T
            dict_task_area[i]=width*height
        
        return dict_boundingBox, dict_task_area

    # @staticmethod
    def generate_convexpoly_for_divided_points(self, dict_dividedPoints, n_cluster, cluster_offset = 0.):
        import shapely.geometry as geom

        dict_boundingBox = {}
        dict_task_area = np.zeros(n_cluster) #define task area based on ROI
        for i in range(n_cluster):            
            broken_sensors = dict_dividedPoints[i][:,:2]
            
            # # Construct convex polygon that cover all points with buffer
            # cvx_hull = geom.MultiPoint(broken_sensors).convex_hull
            # cvx_envelop = cvx_hull.buffer(cluster_offset, join_style=2)

            # Construct convex polygon that cover all points with coverage hole
            row, col = broken_sensors.shape
            coverage_hole_boxes = np.zeros((4*row, col))
            coverage_hole_boxes[:row,:] = broken_sensors + 0.5*self.grid_size*np.array([-1, -1])
            coverage_hole_boxes[row:2*row,:] = broken_sensors + 0.5*self.grid_size*np.array([-1, 1])
            coverage_hole_boxes[2*row:3*row,:] = broken_sensors + 0.5*self.grid_size*np.array([1, 1])
            coverage_hole_boxes[3*row:,:] = broken_sensors + 0.5*self.grid_size*np.array([1, -1])
            cvx_envelop = geom.MultiPoint(coverage_hole_boxes).convex_hull

            # assign exterior to 2d numpy
            vertex_num = len(cvx_envelop.exterior.coords)
            dict_boundingBox[i] = np.zeros((vertex_num,2))
            for j in range(vertex_num):
                dict_boundingBox[i][j] = cvx_envelop.exterior.coords[j]
        
            dict_task_area[i] = cvx_envelop.area

        return dict_boundingBox, dict_task_area

# -----------------------------------------------------------------------------------------------
# Task Allocation algorithm is here
# -----------------------------------------------------------------------------------------------
class TaskAllocation():
    def __init__(self, agents_pos, agents_capabilities, task_pos, task_size):
        # Store agent's information to be allocated
        self._agents_position = agents_pos.copy()
        self._agents_capabilities = agents_capabilities.copy() # Input in area as m^2
        self._agents_num = self._agents_position.shape[0]

        # Store tasks information to be considered
        self._task_pos = task_pos.copy()
        self._task_sizes = task_size.copy()
        self._task_num = len(task_size)

        self._agent_task_dist = self.__calc_agent_task_distance()


    def __calc_agent_task_distance(self):
        agent_task_dist = np.zeros((self._task_num, self._agents_num))
        a = self._task_pos
        b = self._agents_position
        for i in range (self._task_num):
            for j in range (self._agents_num):
                agent_task_dist[i][j] = np.sqrt((a[i,0]-b[j,0])**2+(a[i,1]-b[j,1])**2)
        
        return agent_task_dist
    
    # Maximum and Absolute Value Objective Function
    def lp_allocation(self):
        sensing_range = np.array(self._agents_capabilities)
        cluster_size = np.array(self._task_sizes)

        agent_num = sensing_range.shape[0]
        task_num = len(cluster_size)

        # Create the assignment matrix
        assignment_matrix = np.zeros((agent_num, task_num))
        A1 = np.kron(np.eye(len(sensing_range)), np.ones((1,len(cluster_size))))
        A2 = np.kron(np.ones((1,len(sensing_range))), np.eye(len(cluster_size)))
        b1 = np.ones((1,len(sensing_range))).T
        b2 = np.ones((1,len(cluster_size))).T

        # Calculate the cost matrix based on the difference between sensing_range and cluster_size
        cost_matrix = np.zeros((agent_num, task_num))
        # for i in range(agent_num):
        #     for j in range(task_num):
        #         cost_matrix[i, j] = (cluster_size[j] - (sensing_range[i])) /(sensing_range[i])

        # Cost for Balancing scenario
        for i in range(agent_num):
            for j in range(task_num):
                for k in range(task_num):
                    if j !=k:
                        # cost_matrix[i, j] = (cluster_size[j] - (sensing_range[i])) /(sensing_range[i])
                        cost_matrix[i, j] = ((1 - sensing_range[i] / cluster_size[j]) - ((1 - sensing_range[i] / cluster_size[k]))) 
        c = cost_matrix.flatten()

        model = gp.Model("IP_General_Assignment_Problem_with_Cluster_Allocation")
        model.Params.LogToConsole = 1
        # Create the decision variable
        x = model.addMVar((agent_num*task_num), vtype=GRB.CONTINUOUS, name="x")

        # Create constraints original problem
        model.addConstr(A1 @ x == b1, name="c1")
        model.addConstr(A2 @ x >= b2, name="c2")
    
        model.setObjective(c @ x, gp.GRB.MINIMIZE)

        # Find the optimal solution
        model.optimize()
        model.write("lp_assignment.lp")
        total_runtime = model.Runtime

        # Create the assignment matrix
        x_values = np.round(x.X)
        assignment_matrix = np.zeros((agent_num, task_num))
        for i in range(agent_num):
            for j in range(task_num):
                assignment_matrix[i, j] = x_values[i * task_num + j] * sensing_range[i]
        print(assignment_matrix)

        column_sums = np.sum(assignment_matrix, axis=0)
        print("Column sum:")
        print(column_sums)
        
        # Calculate difference
        difference = cluster_size - column_sums
        print("Difference", difference)
        eval_score = np.maximum(0, difference)
        print("Evaluation score", eval_score)
        total_sum =np.sum(eval_score)
        print("Total sum", total_sum)
        print("Total runtime: ", total_runtime)
        dist_prop = (1 - column_sums / cluster_size)
        print("Distribution Portion: ", dist_prop)

        self.chosenAgentDict = {}
        # Find the column indices with maximum values in each row
        max_indices = np.argmax(assignment_matrix, axis=1)
        # print(max_indices)
        # Iterate through rows and add to the result_dict
        for row_idx, col_idx in enumerate(max_indices):
            if col_idx in self.chosenAgentDict:
                self.chosenAgentDict[col_idx].append(row_idx)
            else:
                self.chosenAgentDict[col_idx] = [row_idx]
        print(self.chosenAgentDict)
        return self.chosenAgentDict

    # Maximum and Absolute Value Objective Function
    def nlp_allocation(self):
        sensing_range = np.array(self._agents_capabilities)
        cluster_size = np.array(self._task_sizes)

        agent_num = sensing_range.shape[0]
        task_num = len(cluster_size)

        # Create the assignment matrix
        assignment_matrix = np.zeros((agent_num, task_num))

        model = gp.Model("NLP_General_Assignment_Problem")
        model.Params.LogToConsole = 1
        model.Params.Presolve = 2
        # Set MIPFocus 3 to get the focus on lower bound
        model.Params.MIPFocus = 3
        model.Params.ImproveStartTime = 1
        # model.Params.RLTCuts = 2
        # model.Params.StrongCGCuts = 2
        model.Params.Cuts = 3
        model.Params.ImproveStartGap  = 0.01

        model.tune()

        # Create the decision variable the delta_ij
        # x_ij = model.addVars(agent_num, task_num, vtype = GRB.BINARY, name = 'x_ij')
        x_ij = model.addMVar((agent_num, task_num), vtype=GRB.BINARY)
        
        # Create constraints
        # model.addConstrs((gp.quicksum(x_ij[i,j] for i in range(agent_num)) >= 1) for j in range(task_num))
        # model.addConstrs((gp.quicksum(x_ij[i,j] for j in range(task_num)) == 1) for i in range(agent_num))
        model.addConstr(x_ij.sum(axis=0) >= 1)
        model.addConstr(x_ij.sum(axis=1) == 1)
        model.update()

        # Constraint with absolute | p_i - p_j |
        # max_p = model.addVar(lb = 0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Diff')
        # min_p = model.addVar(lb = 0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Diff')
        # max_z = model.addVar(lb = 0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name='max')
        # min_z = model.addVar(lb = 0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name='min')
        # obj_func = 0
        
        # for j in range(task_num):
        #     for k in range(task_num):
        #         if j !=k:
        #             pj = (gp.quicksum(sensing_range[i] * x_ij[i,k] / cluster_size[j] for i in range(agent_num)))
        #             pk = (gp.quicksum(sensing_range[i] * x_ij[j,k] / cluster_size[k] for i in range(agent_num)))
        #             # obj_func += np.max(pj) - np.min(pk)

        # model.addConstr(max_p >= pj, name= 'max_p')
        # model.addConstr(max_p >= -pj, name= '-max_p')
        # model.addConstr(min_p <= pk, name= 'min_p')
        # model.addConstr(min_p <= -pk, name= '-min_p')
        # model.addConstr(max_z == max_p, name= 'max_p')
        # model.addConstr(min_z == min_p, name= 'max_p')

        # obj_func = max_z - min_z
        # # model.addGenConstrAbs(abs_value, diff, 'AbsConstr')
        # model.setObjective(obj_func, sense=GRB.MINIMIZE)
        # model.update()


        # Constraints with max(0, objective function)
        aux1 = model.addVars(task_num, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
        aux2 = model.addVars(task_num, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
        # z = model.addVars(task_num, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='z')
        model.update()

        for j in range(task_num):
            model.addConstr(aux1[j] >= cluster_size[j] - gp.quicksum(sensing_range[i] * x_ij[i, j] for i in range(agent_num)))
            model.addConstr(aux1[j] >= 0 )
            model.addGenConstrMax(aux2[j], [aux1[j]], 0.0)
            # New constraint
            lb = -100.0  # Set your lower bound
            ub = 1000.0  # Set your upper bound
            # model.addConstr(lb * (1 - x_ij.sum('*', j)) -aux1[j] <=  - z[j])
            # model.addConstr(aux1[j] - z[j] <= ub * (1 - x_ij.sum('*', j)))
        model.update()
        model.setObjective(gp.quicksum(aux2[j] for j in range(task_num)), sense = GRB.MINIMIZE)

        # Find the optimal solution
        model.optimize()
        # model.write("nlp_assignment.lp")
        total_runtime = model.Runtime

        # Create the assignment matrix
        assignment_matrix = np.zeros((agent_num, task_num))
        for i in range(agent_num):
            for j in range(task_num):
                if np.round(x_ij[i, j].x) == 1:
                    assignment_matrix[i, j] = x_ij[i, j].x * sensing_range[i]
        print(assignment_matrix)
        
        column_sums = np.sum(assignment_matrix, axis=0)
        print("Column sum:")
        print(column_sums)

        # Calculate difference
        difference = cluster_size - column_sums
        print("Difference", difference)
        eval_score = np.maximum(0, difference)
        print("Evaluation score", eval_score)
        total_sum =np.sum(eval_score)
        print("Total sum", total_sum)
        print("Total runtime: ", total_runtime)

        dist_prop = (1 - column_sums / cluster_size)
        print("Distribution Portion: ", dist_prop)

        self.chosenAgentDict = {}
        # Find the column indices with maximum values in each row
        max_indices = np.argmax(assignment_matrix, axis=1)
        # print(max_indices)
        # Iterate through rows and add to the result_dict
        for row_idx, col_idx in enumerate(max_indices):
            if col_idx in self.chosenAgentDict:
                self.chosenAgentDict[col_idx].append(row_idx)
            else:
                self.chosenAgentDict[col_idx] = [row_idx]
        print(self.chosenAgentDict)
        return self.chosenAgentDict


    # Quadratic Value Objective Function
    # --------------------------------------
    def qp_allocation(self):
        # Initial condition
        sensing_range = np.array(self._agents_capabilities)
        cluster_size = np.array(self._task_sizes)

        agent_num = sensing_range.shape[0]
        task_num = len(cluster_size)

        # Create the assignment matrix
        A1 = np.kron(np.eye(len(sensing_range)), np.ones((1,len(cluster_size))))
        A2 = np.kron(np.ones((1,len(sensing_range))), np.eye(len(cluster_size)))
        b1 = np.ones((1,len(sensing_range))).T
        b2 = np.ones((1,len(cluster_size))).T

        # Set parameter for objective function 1
        alpha = 0.005
        q = np.kron(sensing_range.T, np.eye(task_num))
        # Indefinite
        # Q = 2 * (q.T @ q)
        # Negative Definite
        Q = 2 * ( q.T @ q - np.max(np.linalg.eigvals(q.T @ q) + alpha) * np.eye(len(q.T @ q)))
        # Positive Definite
        # Q = -2 * ( q.T @ q - np.max(np.linalg.eigvals(q.T @ q) + alpha) * np.eye(len(q.T @ q)))
        # Q = 2 * ( q.T @ q - np.min(np.linalg.eigvals(q.T @ q) - alpha) * np.eye(len(q.T @ q)))
        p = -2 * (cluster_size.T @ q)
        r = cluster_size @ cluster_size

        # Set parameter for objective function 2
        m_values = len(cluster_size)
        pairs = []
        if m_values > 1:
            for i in range(1, m_values):
                for j in range(i + 1, m_values + 1):
                    pairs.append((i, j))
        else:
            pairs.append((1,1))
        print(f"For m = {m_values}, pairs are: {pairs}")

        matrices_B = []
        matrices_P = []

        for i, j in pairs:
            if m_values > 1:
                matrix_B = np.zeros((m_values, m_values))
                matrix_B[i - 1, i - 1] = 1
                matrix_B[j - 1, j - 1] = 1
                matrices_B.append(matrix_B)

                matrix_P = np.kron(np.outer(sensing_range, sensing_range), matrix_B)
                matrices_P.append(matrix_P)
            else:
                matrix_B = 1
                matrices_B.append(matrix_B)

                matrix_P = np.kron(np.outer(sensing_range, sensing_range), matrix_B)
                matrices_P.append(matrix_P)

        # Sum all P_i matrices to get the final P matrix (ND)
        P = 2*(sum(matrices_P) - (np.max(np.linalg.eigvals(sum(matrices_P))) + alpha) * np.eye(len(sum(matrices_P))))
        # Sum all P_i matrices to get the final P matrix PD
        # P = 2*(sum(matrices_P) - (np.min(np.linalg.eigvals(sum(matrices_P))) - alpha) * np.eye(len(sum(matrices_P))))

        # Set Gurobi model
        model = gp.Model("QuadraticObjective")
        model.Params.LogToConsole = 0
        model.Params.Presolve = 2 # 2 indicates tighter model
        model.Params.MIPFocus = 3
        model.params.Method = 1
        model.params.TuneTimeLimit = 10
        # model.params.BQPcuts = 2
        # model.params.nonConvex = 2 # Apply to handle nonconvex objective function
        model.params.PreQLinearize = 1 # Apply a strong relaxation on QP
        model.tune()
        
        x = model.addMVar((agent_num*task_num), vtype=GRB.CONTINUOUS, name="x")
        model.update()    

        # Model add constraints
        # Add constraints
        model.addConstr(A1 @ x == b1, name="c1")
        model.addConstr(A2 @ x >= b2, name="c2")
        obj1 = 0.5 * x.T @ Q @ x + p.T @ x + r
        obj2 = 0.5 * x.T @ P @ x

        ## Set objective
        # Objective 1 Non overlaping objective
        # model.setObjective(obj1, sense=gp.GRB.MINIMIZE)
        # Objective 2 Balancing objective
        # model.setObjective(obj2, sense=gp.GRB.MINIMIZE)
        # Objective 3  
        # model.setObjective( 0.5 * x.T @ H @ x, sense=gp.GRB.MINIMIZE)
        beta = 0.0
        # Objective 4 for multiobjective
        model.setObjective( beta * obj1 + (1 - beta) * obj2 , sense=gp.GRB.MINIMIZE)
        
        model.update()
        model.optimize()
        model.write("qp_assignment.mps")
        total_runtime = model.Runtime
        
        x_values = (x.X)

        # Code below will show an assignment matrix of drone -> cluster
        assignment_matrix = np.zeros((agent_num, task_num))
        for i in range(agent_num):
            for j in range(task_num):
                assignment_matrix[i, j] = x_values[i * task_num + j] * sensing_range[i]
                    
        print("Assignment matrix \n ", assignment_matrix)

        column_sums = np.sum(assignment_matrix, axis=0)
        print("Column sum:")
        print(column_sums)

        # Calculate difference
        difference = cluster_size - column_sums
        print("Difference", difference)
        eval_score = np.maximum(0, difference)
        print("Evaluation score", eval_score)
        total_sum =np.sum(eval_score)
        print("Total sum", total_sum)
        print("Total runtime: ", total_runtime)
        dist_prop = (1 - column_sums / cluster_size)
        print("Distribution Portion: ", dist_prop)

        # print("Eig Q: ", np.linalg.eigvals(Q))
        # print("Eig H: ", np.linalg.eigvals(P))
        print("Optimal value", x_values)

        self.chosenAgentDict = {}
        # Find the column indices with maximum values in each row
        max_indices = np.argmax(assignment_matrix, axis=1)
        # print(max_indices)
        # Iterate through rows and add to the result_dict
        for row_idx, col_idx in enumerate(max_indices):
            if col_idx in self.chosenAgentDict:
                self.chosenAgentDict[col_idx].append(row_idx)
            else:
                self.chosenAgentDict[col_idx] = [row_idx]
        print(self.chosenAgentDict)
        return self.chosenAgentDict
    
    def tchebycheff_allocation(self, weights=None):
        sensing_range = np.array(self._agents_capabilities)
        cluster_size = np.array(self._task_sizes)
        agent_num = sensing_range.shape[0]
        task_num = len(cluster_size)

        # Assignment matrices and constraints
        A1 = np.kron(np.eye(agent_num), np.ones((1, task_num)))
        A2 = np.kron(np.ones((1, agent_num)), np.eye(task_num))
        b1 = np.ones((agent_num, 1))
        b2 = np.ones((task_num, 1))

        # Objective function parameters
        alpha = 0.005
        q = np.kron(sensing_range.T, np.eye(task_num))
        Q = 2 * (q.T @ q - np.max(np.linalg.eigvals(q.T @ q) + alpha) * np.eye(len(q.T @ q)))
        p = -2 * (cluster_size.T @ q)
        r = cluster_size @ cluster_size

        # Set parameter for objective function 2
        m_values = len(cluster_size)
        pairs = []
        if m_values > 1:
            for i in range(1, m_values):
                for j in range(i + 1, m_values + 1):
                    pairs.append((i, j))
        else:
            pairs.append((1,1))
        print(f"For m = {m_values}, pairs are: {pairs}")

        matrices_B = []
        matrices_P = []

        for i, j in pairs:
            if m_values > 1:
                matrix_B = np.zeros((m_values, m_values))
                matrix_B[i - 1, i - 1] = 1
                matrix_B[j - 1, j - 1] = 1
                matrices_B.append(matrix_B)

                matrix_P = np.kron(np.outer(sensing_range, sensing_range), matrix_B)
                matrices_P.append(matrix_P)
            else:
                matrix_B = 1
                matrices_B.append(matrix_B)

                matrix_P = np.kron(np.outer(sensing_range, sensing_range), matrix_B)
                matrices_P.append(matrix_P)

        # Sum all P_i matrices to get the final P matrix (ND)
        P = 2*(sum(matrices_P) - (np.max(np.linalg.eigvals(sum(matrices_P))) + alpha) * np.eye(len(sum(matrices_P))))
        # Sum all P_i matrices to get the final P matrix PD
        # P = 2*(sum(matrices_P) - (np.min(np.linalg.eigvals(sum(matrices_P))) - alpha) * np.eye(len(sum(matrices_P))))

        # Gurobi model setup for baseline objective values
        model = gp.Model("TchebycheffScalarization")
        model.Params.LogToConsole = 0
        x = model.addMVar((agent_num * task_num), vtype=GRB.CONTINUOUS, name="x")

        model.addConstr(A1 @ x == b1.flatten(), name="c1")
        model.addConstr(A2 @ x >= b2.flatten(), name="c2")

        # Baseline optimization for obj1 and obj2
        model.setObjective(0.5 * x @ Q @ x + p @ x + r, GRB.MINIMIZE)
        model.optimize()
        baseline_obj1 = model.ObjVal

        model.setObjective(0.5 * x @ P @ x, GRB.MINIMIZE)
        model.optimize()
        baseline_obj2 = model.ObjVal

        # Tchebycheff scalarization
        if weights is None:
            weights = [1.0, 0.0]

        # model.setObjective(None)
        max_deviation = model.addVar(vtype=GRB.CONTINUOUS, name="max_deviation")
        model.addConstr(weights[0] * (0.5 * x @ Q @ x + p @ x + r - baseline_obj1) <= max_deviation, name="obj1_dev")
        model.addConstr(weights[1] * (0.5 * x @ P @ x - baseline_obj2) <= max_deviation, name="obj2_dev")

        model.setObjective(max_deviation, GRB.MINIMIZE)
        model.optimize()

        # Process results
        x_values = x.X
        assignment_matrix = np.zeros((agent_num, task_num))
        for i in range(agent_num):
            for j in range(task_num):
                assignment_matrix[i, j] = x_values[i * task_num + j] * sensing_range[i]

        print("Assignment matrix \n", assignment_matrix)
        column_sums = np.sum(assignment_matrix, axis=0)
        difference = cluster_size - column_sums
        eval_score = np.maximum(0, difference)
        total_sum = np.sum(eval_score)
        dist_prop = (1 - column_sums / cluster_size)

        print("Column sum:", column_sums)
        print("Difference", difference)
        print("Evaluation score", eval_score)
        print("Total sum", total_sum)
        print("Distribution Portion:", dist_prop)
        print("Optimal value", x_values)

        max_indices = np.argmax(assignment_matrix, axis=1)
        for row_idx, col_idx in enumerate(max_indices):
            if col_idx in self.chosenAgentDict:
                self.chosenAgentDict[col_idx].append(row_idx)
            else:
                self.chosenAgentDict[col_idx] = [row_idx]
        print("Chosen Agent Dictionary:", self.chosenAgentDict)
        return self.chosenAgentDict

    def multiobjective_allocation(self):
        sensing_range = np.array(self._agents_capabilities)
        cluster_size = np.array(self._task_sizes)

        agent_num = sensing_range.shape[0]
        task_num = len(cluster_size)

        # Set parameter for objective function 1
        alpha = 0.005
        q = np.kron(sensing_range.T, np.eye(task_num))
        # Positive Definite
        Q = 2 * ( q.T @ q - np.min(np.linalg.eigvals(q.T @ q) - alpha) * np.eye(len(q.T @ q)))
        p = -2 * (cluster_size.T @ q)
        r = cluster_size @ cluster_size

        # Set parameter for objective function 2
        m_values = len(cluster_size)
        pairs = []
        if m_values > 1:
            for i in range(1, m_values):
                for j in range(i + 1, m_values + 1):
                    pairs.append((i, j))
        else:
            pairs.append((1,1))
        print(f"For m = {m_values}, pairs are: {pairs}")

        matrices_B = []
        matrices_P = []

        for i, j in pairs:
            if m_values > 1:
                matrix_B = np.zeros((m_values, m_values))
                matrix_B[i - 1, i - 1] = 1
                matrix_B[j - 1, j - 1] = 1
                matrices_B.append(matrix_B)

                matrix_P = np.kron(np.outer(sensing_range, sensing_range), matrix_B)
                matrices_P.append(matrix_P)
            else:
                matrix_B = 1
                matrices_B.append(matrix_B)

                matrix_P = np.kron(np.outer(sensing_range, sensing_range), matrix_B)
                matrices_P.append(matrix_P)

        # Sum all P_i matrices to get the final P matrix (ND)
        # P = 2*(sum(matrices_P) - (np.max(np.linalg.eigvals(sum(matrices_P))) + alpha) * np.eye(len(sum(matrices_P))))
        # Sum all P_i matrices to get the final P matrix PD
        P =  2*(sum(matrices_P) - (np.min(np.linalg.eigvals(sum(matrices_P))) - alpha) * np.eye(len(sum(matrices_P))))

        n = len(sensing_range) * len(cluster_size)
        # Define variables
        x = cp.Variable(n, boolean=True)
        # Define new RLT variables y_ij = x_i * x_j
        y = cp.Variable((n, n), symmetric=True)

        # Define constraints
        constraints = []

        A1 = np.kron(np.eye(len(sensing_range)), np.ones((1, len(cluster_size))))
        A2 = np.kron(np.ones((1, len(sensing_range))), np.eye(len(cluster_size)))
        b1 = np.ones((1, len(sensing_range))).T
        b2 = np.ones((1, len(cluster_size))).T

        constraints.append(A1 @ x == b1.flatten())
        constraints.append(A2 @ x >= b2.flatten())
        constraints.append(0 <= x)
        constraints.append(x <= 1)

        # Add RLT constraints
        for i in range(n):
            for j in range(i, n):
                constraints.append(y[i, j] >= 0)                  # y_ij >= 0
                constraints.append(y[i, j] <= x[i])               # y_ij <= x_i
                constraints.append(y[i, j] <= x[j])               # y_ij <= x_j
                constraints.append(y[i, j] >= x[i] + x[j] - 1)    # y_ij >= x_i + x_j - 1

        # Define objective functions
        f1 = cp.real(0.5 * cp.sum(cp.multiply(Q, y)) + p @ x + r)
        f2 = cp.real(0.5 * cp.sum(cp.multiply(P, y))) 

        # Parameter for multiobjective function
        lam = 0.5
        # Scalarized objective
        obj = (1 - lam) * f1 + lam * f2

        start_time = time.time()
        # Solve the problem
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.SCIP)
        end_time = start_time - time.time()
        print("MOO time: ", end_time)

        x_values = x.value
        # Code below will show an assignment matrix of drone -> cluster
        assignment_matrix = np.zeros((agent_num, task_num))
        for i in range(agent_num):
            for j in range(task_num):
                assignment_matrix[i, j] = x_values[i * task_num + j] * sensing_range[i]
                    
        print("Assignment matrix \n ", assignment_matrix)

        column_sums = np.sum(assignment_matrix, axis=0)
        print("Column sum:")
        print(column_sums)

        self.chosenAgentDict = {}
        # Find the column indices with maximum values in each row
        max_indices = np.argmax(assignment_matrix, axis=1)
        # print(max_indices)
        # Iterate through rows and add to the result_dict
        for row_idx, col_idx in enumerate(max_indices):
            if col_idx in self.chosenAgentDict:
                self.chosenAgentDict[col_idx].append(row_idx)
            else:
                self.chosenAgentDict[col_idx] = [row_idx]
        print(self.chosenAgentDict)
        return self.chosenAgentDict

    # GREEDY ALLOCATION
    # --------------------------------------
    def allocate_greedy(self):
        choosable_agents = np.ones(self._agents_num) #initially, all agents can be chosen by the task
        chosen_agent = np.zeros((self._task_num, int(self._agents_num/self._task_num)))

        for i in range (self._task_num):
            for k in range (int(self._agents_num/self._task_num)):
                for j in range (self._agents_num):
                    if choosable_agents[j]==0:
                        self._agent_task_dist[i,j] = 10000 #set as large as possible so that this will not be chosen
                min = np.min(self._agent_task_dist[i])
                condition = (self._agent_task_dist[i] == min)
                result = np.where(condition)
                #print(result[0])
                chosen_agent[i,k]=result[0]
                choosable_agents[result[0]]=0
        #print(chosen_agent)

        # Implement chosen agent as dictionary
        chosen_agent_idx = chosen_agent.astype(int).tolist()
        self.chosenAgentDict = {}
        for i in range (self._task_num):
            self.chosenAgentDict[i] = chosen_agent_idx[i]
            print(self.chosenAgentDict)
        return self.chosenAgentDict


    # ACO static functions
    # --------------------------------------
    @staticmethod
    def heuristic(task, agent, agent_task_dist, chosen_agent, agent_sensing, task_area):
        sensing_sum = 0
        for i in range (len(agent_sensing)): #size of agent
            if (i+1) in chosen_agent[task]: 
            #note that the chosen agent numbers starts from 1 instead of 0
            #note that each task can choose any numbers of agents
                sensing_sum += agent_sensing[i]
        func = (task_area[task]-sensing_sum)/agent_task_dist[task][agent]
        return func
    
    @staticmethod
    def efficiency(chosen_agent, agent_task_dist, agent_sensing, task_area):
        num = 0 #initialization
        denum = 0
        for i in range (agent_task_dist.shape[0]): #size of task
            sensing_sum = 0
            for j in range (agent_task_dist.shape[1]): #size of agent
                if (j+1) in chosen_agent[i]: 
                #note that the chosen agent numbers starts from 1 instead of 0
                #note that each task can choose any numbers of agents
                    sensing_sum += agent_sensing[j]
                    denum += agent_task_dist[i,j]
            num += task_area[i]-abs(sensing_sum-task_area[i])  
            # Default num/denum
        return num
    # --------------------------------------

    def allocate_MACO(self):
        start_time = time.time()
        #initialization of parameters
        alpha = 1 #relative importance of the pheromone
        beta = 1 #relative importance of the heuristic function
        rho = 0.2 #pheromone evaporation coefficient 

        m_ants = 10 #number of ants
        max_iter = 1000 #number of max iteration
        fero_init = 1 #initial pheromone value at each edge
        fero_str = 1.0 #pheromone strength, Q

        fero = np.ones((self._task_num, self._agents_num))*fero_init
        fero_update = np.zeros((self._task_num, self._agents_num))
        choseniter = np.zeros((m_ants, self._task_num, self._agents_num)) #to keep the info of ant tour at the end of iteration
        effiter = np.zeros(m_ants) #to keep the info of efficiency at the end of iteration

        for iter in range (max_iter):
            for i in range (m_ants):
                #define agents that can be chosen
                choosable_tasks = np.ones(self._task_num) #initially, all tasks can be chosen by the ants
                choosable_agents = np.ones(self._agents_num) #initially, all agents can be chosen by the ants
                chosen_agent = np.zeros((self._task_num, self._agents_num)) #initialization, no agent is chosen yet

                for j in range (self._agents_num): #penentuan pasangan task-agent dilakukan sebanyak jumlah agent
                    probability = np.zeros(self._task_num*self._agents_num) #initialization of probability of each agent
                    prob_sum = 0
                    count = 0
                    for k in range (self._task_num):
                        for l in range (self._agents_num):
                        #calculate probability of each task and agent for that particular couple
                            if (int(choosable_tasks[k])==1 and int(choosable_agents[l])==1): 
                                #if the task and the agent can both be chosen
                                probability[count] = (fero[k,l]**alpha)*(
                                    self.heuristic(k,l,self._agent_task_dist,chosen_agent,self._agents_capabilities,self._task_sizes) **beta)

                                ###------------additional code 1 by Lisa ----------------------
                                if (probability[count] < 0): probability[count] = 0
                                ###------------end of additional code 1----------------------
                            count += 1
                    prob_sum = sum(probability)
                    # if (prob_sum == 0): break
                    ###------------additional code 2 by Lisa ----------------------
                    if (prob_sum <= 0):
                        if not is_all_tasks_assigned:
                        # if there are still tasks that has not been given any agent, repeat probability calculation
                        # all tasks have equal probability to be paired with agent, heuristic value = 1
                            count = 0
                            for k in range (self._task_num):
                                for l in range (self._agents_num):
                                #re-calculate probability of each task and agent for that particular couple
                                    if (int(choosable_tasks[k])==1 and int(choosable_agents[l])==1): 
                                        #if the task and the agent can both be chosen
                                        probability[count] = (fero[k,l]**alpha)*1
                                    count += 1
                            prob_sum = sum(probability)
                        else:
                            break
                    ###------------end of additional code 2----------------------
                    probability = probability/prob_sum
                    #print(probability)

                    agent_task = np.random.choice(self._agents_num*self._task_num, 1, p=probability) + 1
                    #roullete wheel for choosing the agents and task, note that it starts from 0
                    #print(agent_task)

                    task = int(agent_task // self._agents_num)
                    agent = int(agent_task % self._agents_num)-1
                    if agent<0:
                        agent = agent+self._agents_num
                        task = task-1
                    #print(task)
                    #print(agent)

                    chosen_agent[task][agent] = agent+1 #note that the agent calculation starts from 1
                    choosable_agents[agent] = 0 #agent that is already chosen cannot be chosen anymore (one-to-one)

                    # ORIGINAL CODE FROM LISA
                    # ------------------------------------------------------------------------------------
                    # #check if the task can still be chosen or not
                    # for m in range (self._task_num):
                    #     agent_sensing_sum = 0
                    #     for n in range (self._agents_num): #size of agent
                    #         if (n+1) in chosen_agent[m]: 
                    #             agent_sensing_sum += self._agents_capabilities[n]
                    #     if (self._task_sizes[m] <= agent_sensing_sum):
                    #         choosable_tasks[m] = 0 #task that already have enough agent cannot be chosen anymore

                    # WIDHI'S EDIT BASED ON DISCUSSION WITH LISA
                    # ------------------------------------------------------------------------------------
                    #check if the task can still be chosen or not
                    is_task_m_assigned_yet = [False]*self._task_num
                    agent_m_sensing_sum = [0]*self._task_num
                    # Check if each task is assigned and compute the current sensing sum
                    for m in range (self._task_num):
                        for n in range (self._agents_num): #size of agent
                            if (n+1) in chosen_agent[m]: 
                                is_task_m_assigned_yet[m] = True # flagged that this task is assigned at minimum 1 agent
                                agent_m_sensing_sum[m] += self._agents_capabilities[n]
                    # Re-evaluate the choosable task
                    is_all_tasks_assigned = (sum(is_task_m_assigned_yet) == self._task_num)
                    for m in range (self._task_num):
                        if not is_all_tasks_assigned:
                            if is_task_m_assigned_yet[m]: 
                                choosable_tasks[m] = 0 #chosen task cannot be chosen anymore, until all tasks chosen
                        else:
                            if (self._task_sizes[m] <= agent_m_sensing_sum[m]):
                                choosable_tasks[m] = 0 #task that already have enough agent cannot be chosen anymore
                            else: choosable_tasks[m] = 1 # Reset from the previous case when only some tasks are assigned


                #Saving the pheromone update by one ant
                #Will be updated after one iteration
                for m in range (self._task_num):
                    for n in range (self._agents_num):
                        if chosen_agent[m,n]!=0:
                            fero_update[m,n] += fero_str*self.efficiency(chosen_agent, self._agent_task_dist, self._agents_capabilities, self._task_sizes)
                            #pheromone is only updated if the path is chosen by the ant
                if iter==max_iter-1:
                #When the iteration is finished:
                #save the efficiency of each ant and the route of each ants
                #to select the best agent-task allocation (max eff)
                    #print(iter)
                    choseniter[i] = chosen_agent
                    #print("ants no:", i)
                    #print(choseniter[i])
                    effiter[i] = self.efficiency(chosen_agent, self._agent_task_dist, self._agents_capabilities, self._task_sizes)
                    #print(effiter[i])

            #Pheromone update after one iteration
            fero = (1-rho)*fero+fero_update

        max_eff = np.max(effiter)
        id = np.where(effiter==max_eff)
        chosen_conf = choseniter[id] 
        #print('Ant no : ' +str(id[0]))
        #print('Maximum efficiency : ' +str(max_eff))
        #print('Chosen configuration : ')
        #print('(note that the rows represent the tasks and the collumns represent the agents)')
        #print('(chosen agents are marked with the non-zero number)')
        #print(chosen_conf[0]) #only show the first configuration with max efficiency
        chosen_final = chosen_conf[0] 
        #print("Agent sensing area :")
        #print(self.sensingAreaList)
        #print("Task area :")
        #print(self.task_area)


        self.chosenAgentDict={}
        for i in range (self._task_num):
            self.chosenAgentDict[i] = np.where(chosen_final[i] > 0)[0].astype(int).tolist()
            # print(self.chosenAgentDict)
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
        return self.chosenAgentDict



class EvaluateAllocation():
    def __init__(self, broken_sensor_pos, dict_dividedPoints, dict_boundingBox, task_pos, task_size,
                agents_pos, agents_capabilities, chosenAgentIdx ): 
        # Store scenarios data
        self._broken_sensor_pos = broken_sensor_pos.copy()
        self._dict_dividedPoints = dict_dividedPoints
        self._dict_boundingBox = dict_boundingBox

        # Store agent's information to be allocated
        self._agents_position = agents_pos.copy()
        self._agents_capabilities = agents_capabilities.copy() # Input in area as m^2
        self._agents_num = self._agents_position.shape[0]

        # Store tasks information to be considered
        self._task_pos = task_pos.copy()
        self._task_sizes = task_size.copy()
        self._task_num = len(task_size)
        
        # Store allocation results
        self._chosenAgentIdx = chosenAgentIdx        


    def print_allocation_info(self):
        print('Total number of malfunctioned sensors: ')
        print(self._broken_sensor_pos.size)

        print('Agents allocated')
        print(self._chosenAgentIdx)

        for i in range(self._task_num):
            print('----------NEW CLUSTER ------------')
            print(self._agents_position[self._chosenAgentIdx[i], 0:2])
            print('bounding box')
            print(self._dict_boundingBox[i].T)
            print('broken sensors')
            print(self._dict_dividedPoints[i].T)
            print('centroid')
            cent = np.array([self._task_pos[i]])
            print(cent)


    def print_agents_task_dist(self):
        agent_task_dist = np.zeros((self._task_num, self._agents_num))
        a = self._task_pos
        b = self._agents_position
        for i in range (self._task_num):
            for j in range (self._agents_num):
                agent_task_dist[i][j] = np.sqrt((a[i,0]-b[j,0])**2+(a[i,1]-b[j,1])**2)

        #Evaluation
        #Average UAV-task distance
        #Average difference of task area and sensing area 
        Av_dist = 0
        Av_area_diff = 0
        count = 0
        for i in range (self._task_num):
            Agent_sense_sum = 0
            for j in self._chosenAgentIdx[i]:
                Av_dist += agent_task_dist[i,j]
                Agent_sense_sum += self._agents_capabilities[j]
                count +=1  #note that thare may be cases where not all agents are selected 
            Av_area_diff += abs(Agent_sense_sum-self._task_sizes[i])
        print('Average UAV-task distance :')
        print(Av_dist/count)
        print('Average difference of task area and sensing area :')
        print(Av_area_diff/self._task_num)    


    def plot_cluster(self, fname=None):
        fig = plt.figure()
        # plt.rcParams.update({'font.size': 14})
        # plt.axis('scaled')
        plt.gca().set_aspect('equal', adjustable='box', anchor='C')
        #ax = fig.add_axes([1,1,1,1])
        #Plotting the clustered data
        # colorList = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        colorList = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Draw all broken sensors in uniform gray square
        plt.scatter(self._broken_sensor_pos[:,0], self._broken_sensor_pos[:,1], 
                    s = 140, marker = 's', color = '0.5', label='Broken Sensors')
        # Draw the polygon
        for i in range(self._task_num):
            x, y = self._task_pos[i,0], self._task_pos[i,1]
            plt.scatter(x, y, s = 40, color = colorList[i], label='Centroid $c^'+str(i+1)+'$' ) #Plotting the centroid / task
            plt.plot(self._dict_boundingBox[i][:,0], self._dict_boundingBox[i][:,1], '-', color = colorList[i], label='Region $\mathcal{Q}^'+str(i+1)+'$')

        # # Draw all broken sensors in each cluster color
        # for i, clusterPosArray in self._dict_dividedPoints.items():
        #     plt.scatter(clusterPosArray[:,0], clusterPosArray[:,1], color = colorList[i], label='Cluster '+str(i+1))
        # #Plotting the centroids
        # plt.scatter(self._task_pos[:,0], self._task_pos[:,1], s = 50, marker = 's', color = 'k', label='Centroids')

        # plt.xticks([ 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20 ])
        # plt.yticks([ 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20 ])
        # plt.xticks([ 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20 ], [ 0, .5, 1., 1.5, 2., 2.5, 3., 3.5, 4 ])
        # plt.yticks([ 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20 ], [ 0, .5, 1., 1.5, 2., 2.5, 3., 3.5, 4 ])
        plt.xticks([ -2, -1.75, -1.25, -0.75, 0., 0.75, 1.25, 1.75, 2 ])
        plt.yticks([ 2, -1.75, -1.25, -0.75, 0., 0.75, 1.25, 1.75, 2 ])
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        # plt.grid(True)
        # plt.title('Clustered Broken Sensors and Their Centroids')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        # plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

        if fname is None:
            plt.show(block=False)
        else: # save to fname
            plt.savefig(fname, bbox_inches="tight")
            plt.close(fig)

        # Revert the default font size as to not mess up with the other figure
        # plt.rcParams.update({'font.size': 10})


    def plot_allocation(self, fname=None):
        #plotting the agent and the tasks - 2
        # colorList = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        colorList = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig = plt.figure()
        # plt.rcParams.update({'font.size': 14})
        # plt.axis('scaled')
        plt.gca().set_aspect('equal', adjustable='box', anchor='C')

        n_agents = len(self._agents_position)
        agentColor = ['k']*n_agents

        text_offset = 0.5 #*self.grid_size
        for i in range (self._task_pos.shape[0]):
            x, y = self._task_pos[i,0], self._task_pos[i,1]
            plt.scatter(x, y, s = 40, color = colorList[i], label='Allocation '+str(i+1) ) #Plotting the centroid / task
            # plt.text(x-5*text_offset, y+2*text_offset, "{:.2f}".format(self._task_sizes[i]), color = colorList[i], fontsize='large')
            
            # Reallocate color for agents
            for a in self._chosenAgentIdx[i]:
                agentColor[a] = colorList[i]
            
                # Draw arrow to the task
                arrow_par = 0.5
                dx, dy = x - self._agents_position[a,0], y - self._agents_position[a,1]
                dist = np.linalg.norm([dx, dy])
                k = (dist - arrow_par - 0.2) / dist
                plt.arrow(self._agents_position[a,0], self._agents_position[a,1], k*dx, k*dy, 
                          linestyle = '-', head_width=arrow_par, head_length=arrow_par, color = colorList[i])

        for i in range (n_agents):
            x = self._agents_position[i,0]
            y = self._agents_position[i,1]
            plt.scatter(x, y, marker = 'X', s = 80, color = colorList[i])
            # plt.text(x+text_offset, y+text_offset, str(self._agents_capabilities[i]), color = agentColor[i], fontsize='large')


        # plt.xticks([ 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20 ])
        # plt.yticks([ 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20 ])
        # plt.xticks([ 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20 ], [ 0, .5, 1., 1.5, 2., 2.5, 3., 3.5, 4 ])
        # plt.yticks([ 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20 ], [ 0, .5, 1., 1.5, 2., 2.5, 3., 3.5, 4 ])
        plt.xticks([ -2, -1.75, -1.25, -0.75, 0., 0.75, 1.25, 1.75, 2 ])
        plt.yticks([ 2, -1.75, -1.25, -0.75, 0., 0.75, 1.25, 1.75, 2 ])
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        # plt.grid(True)
        plt.title('Tasks Allocation')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        # plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

        if fname is None:
            plt.show(block=False)
        else: # save to fname
            plt.savefig(fname, bbox_inches="tight")
            plt.close(fig)
        
        # Revert the default font size as to not mess up with the other figure
        # plt.rcParams.update({'font.size': 10})


    def plot_clusterAllocate(self, fname=None):
        fig = plt.figure()
        plt.rcParams.update({'font.size': 14})
        plt.rcParams['text.usetex'] = True
        plt.axis('scaled')
        #ax = fig.add_axes([1,1,1,1])
        #Plotting the clustered data
        # colorList = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        colorList = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # n_agents = len(agents_position)
        # for i in range(self.clusterNum):
        #     clusterPosArray = self.dividedPoints[i]
        #     plt.scatter(clusterPosArray[:,0], clusterPosArray[:,1], color = 'k', label='Sensors'+str(i+1))
        
        plt.scatter(self._broken_sensor_pos[:,0], self._broken_sensor_pos[:,1], color = 'k', label='Broken Static \nSensors')

        for i in range(self._task_num):
            x, y = self._task_pos[i,0], self._task_pos[i,1]
            plt.scatter(x, y, s = 100, marker = 's', color = colorList[i], label='Centroid $c^'+str(i+1)+'$' ) #Plotting the centroid / task
            plt.plot(self._dict_boundingBox[i][:,0], self._dict_boundingBox[i][:,1], '-', color = colorList[i], label='Region $\mathcal{Q}^'+str(i+1)+'$')

            #plt.text(text_px[i], text_py[i], str(task_area[i]), color = colorList[i], fontsize='large')

            # Reallocate color for agents
            PosX = []
            PosY = []
            agentColor = None
            for a in self._chosenAgentIdx[i]:
                agentColor = colorList[i]
                PosX += [self._agents_position[a,0]]
                PosY += [self._agents_position[a,1]]

            plt.scatter(PosX, PosY, s=100, marker='X', color = agentColor, label='Quadrotor $i \in \mathcal{I}^'+str(i+1)+'$')

        #for i in range (n_agents):
        #    x = self.agentsPositions[i,0]
        #    y = self.agentsPositions[i,1]
        #    plt.plot(x, y, marker='X', markersize=10, color = agentColor[i], label='drone $i \in \mathcal{I}^'+str(i+1)+'$')

        #plt.xticks([-5, 0, 5, 10, 15, 20, 25, 30])
        #plt.yticks([-5, 0, 5, 10, 15, 20, 25])
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.grid(True)
        plt.title('Broken Static Sensors and Task Allocation')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        
        # Reordering the label
        # handles, labels = plt.gca().get_legend_handles_labels()
        #order = [3, 0,4,5, 1,6,7, 2,8,9 ]
        # order = [3, 0,1,2, 4,6,8, 5,7,9 ]
        # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], bbox_to_anchor=(1.04,1), loc="upper left")        
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

        if fname is None:
            plt.show(block=False)
        else: # save to fname
            plt.savefig(fname, bbox_inches="tight")
            plt.close(fig)

        # Revert the default font size as to not mess up with the other figure
        plt.rcParams.update({'font.size': 10})
        plt.rcParams['text.usetex'] = False