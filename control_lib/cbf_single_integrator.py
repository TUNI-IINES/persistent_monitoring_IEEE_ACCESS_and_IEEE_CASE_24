import numpy as np
import cvxopt

class cbf_si():
    def __init__(self, P = None, q = None, scale_constraint=False):
        self.is_scale_constraint = scale_constraint
        self.reset_cbf()

    def reset_cbf(self):
        # initialize G and h, Then fill it afterwards
        self.constraint_G = None
        self.constraint_h = None
        self.cbf_values = None

    def __set_constraint(self, G_mat, h_mat):
        if self.constraint_G is None:
            self.constraint_G = G_mat
            self.constraint_h = h_mat
        else:
            self.constraint_G = np.append(self.constraint_G, G_mat, axis=0)
            self.constraint_h = np.append(self.constraint_h, h_mat, axis=0)

    ## Uncommented below to use the original
    # def compute_safe_controller(self, u_nom, P = None, q = None):

    #     if (P is None) and (q is None): P, q = 2*np.eye(3), -2*u_nom
        
    #     if self.constraint_G is not None:
    #         G_mat = self.constraint_G.copy()
    #         h_mat = self.constraint_h.copy()
    #         # IMPLEMENTATION OF Control Barrier Function
    #         if self.is_scale_constraint:
    #             for i in range(len(h_mat)): 
    #                 G_mat[i] = self.constraint_G[i] / self.constraint_h[i]
    #                 h_mat[i] = 1.

    #         # Minimization
    #         P_mat = cvxopt.matrix( P.astype(np.double), tc='d')
    #         q_mat = cvxopt.matrix( q.astype(np.double), tc='d')
    #         # Resize the G and H into appropriate matrix for optimization
    #         G_mat = cvxopt.matrix( G_mat.astype(np.double), tc='d') 
    #         h_mat = cvxopt.matrix( h_mat.astype(np.double), tc='d')

    #         # Solving Optimization
    #         cvxopt.solvers.options['show_progress'] = False
    #         sol = cvxopt.solvers.qp(P_mat, q_mat, G_mat, h_mat, verbose=False)

    #         if sol['status'] == 'optimal':
    #             # Get solution + converting from cvxopt base matrix to numpy array
    #             u_star = np.array([sol['x'][0], sol['x'][1], sol['x'][2]])
    #         else: 
    #             print( 'WARNING QP SOLVER' + ' status: ' + sol['status'] + ' --> use nominal instead' )
    #             print(G_mat)
    #             print(h_mat)
    #             u_star = u_nom.copy()
    #     else: # No constraints imposed
    #         u_star = u_nom.copy()

    #     return u_star

    # This is modification to avoid the rank(A) error 
    def compute_safe_controller(self, u_nom, P=None, q=None):
        if (P is None) and (q is None):
            P, q = 2*np.eye(3), -2*u_nom

        # Regularize P matrix
        epsilon = 1e-6
        P += epsilon * np.eye(P.shape[0])

        if self.constraint_G is not None:
            G_mat = self.constraint_G.copy()
            h_mat = self.constraint_h.copy()

            # Filter out constraints with very small values
            valid_constraints = np.abs(h_mat) > 1e-10
            if not np.all(valid_constraints):
                G_mat = G_mat[valid_constraints.flatten()]
                h_mat = h_mat[valid_constraints]

            if self.is_scale_constraint and len(h_mat) > 0:
                G_scaled = np.zeros_like(G_mat)
                h_scaled = np.ones_like(h_mat)
                for i in range(len(h_mat)):
                    if abs(h_mat[i]) > 1e-6:
                        G_scaled[i] = G_mat[i] / h_mat[i]
                G_mat = G_scaled
                h_mat = h_scaled

            try:
                # Convert to cvxopt matrices with explicit type specification
                P_mat = cvxopt.matrix(P.astype(np.double), tc='d')
                q_mat = cvxopt.matrix(q.astype(np.double), tc='d')
                G_mat = cvxopt.matrix(G_mat.astype(np.double), tc='d')
                h_mat = cvxopt.matrix(h_mat.astype(np.double), tc='d')

                # Set solver options
                cvxopt.solvers.options['show_progress'] = False
                cvxopt.solvers.options['abstol'] = 1e-8
                cvxopt.solvers.options['reltol'] = 1e-7
                cvxopt.solvers.options['feastol'] = 1e-7

                # Solve QP problem
                sol = cvxopt.solvers.qp(P_mat, q_mat, G_mat, h_mat, verbose=False)

                if sol['status'] == 'optimal':
                    u_star = np.array(sol['x']).flatten()
                else:
                    print(f"Warning: QP solver failed with status {sol['status']}")
                    print("Falling back to nominal control")
                    u_star = u_nom.copy()
            except Exception as e:
                print(f"QP solver error: {str(e)}")
                print("Falling back to nominal control")
                u_star = u_nom.copy()
        else:
            u_star = u_nom.copy()

        return u_star



    # ADDITION OF CONSTRAINTS
    # -----------------------------------------------------------------------------------------------------------
    def add_avoid_static_circle(self, pos, obs, ds, gamma=10, power=3):
        # h = norm2( pos - obs )^2 - norm2(ds)^2 > 0
        vect = pos - obs
        h_func = np.power(np.linalg.norm(vect), 2) - np.power(ds, 2)
        # -(dh/dpos)^T u < gamma(h)
        self.__set_constraint(-2*vect.reshape((1,3)), gamma*np.power(h_func, power).reshape((1,1)))

        return h_func


    def add_maintain_distance_with_epsilon(self, pos, obs, ds, epsilon, gamma=10, power=3):
        vect = pos - obs
        # h = norm2( ds + epsilon )^2 - norm2( pos - obs )^2 > 0
        h_func_l = np.power((ds+epsilon), 2) - np.power(np.linalg.norm(vect), 2)
        # -(dh/dpos)^T u < gamma(h)
        self.__set_constraint(2*vect.reshape((1,3)), gamma*np.power(h_func_l, power).reshape((1,1)))

        # h = norm2( pos - obs )^2 - norm2( ds - epsilon )^2 > 0
        h_func_u = np.power(np.linalg.norm(vect), 2) - np.power((ds-epsilon), 2)
        # -(dh/dpos)^T u < gamma(h)
        self.__set_constraint(-2*vect.reshape((1,3)), gamma*np.power(h_func_u, power).reshape((1,1)))

        return h_func_l, h_func_u


    def add_avoid_static_ellipse(self, pos, obs, theta, major_l, minor_l, gamma=10, power=3):
        # h = norm2( ellipse*[pos - obs] )^2 - 1 > 0
        theta = theta if np.ndim(theta) == 0 else theta.item()
        # TODO: assert a should be larger than b (length of major axis vs minor axis)
        vect = pos - obs # compute vector towards pos from centroid
        # rotate vector by -theta (counter the ellipse angle)
        # then skew the field due to ellipse major and minor axis
        # the resulting vector should be grater than 1
        # i.e. T(skew)*R(-theta)*vec --> then compute L2norm square
        ellipse = np.array([[2./major_l, 0, 0], [0, 2./minor_l, 0], [0, 0, 1]]) \
            @ np.array([[np.cos(-theta), -np.sin(-theta), 0], [np.sin(-theta), np.cos(-theta), 0], [0, 0, 1]], dtype=object)
        h_func = np.power(np.linalg.norm( ellipse @ vect.T ), 2) - 1
        # -(dh/dpos)^T u < gamma(h)
        # -(2 vect^T ellipse^T ellipse) u < gamma(h)
        G = -2*vect @ ( ellipse.T @ ellipse )
        self.__set_constraint( G.reshape((1,3)), gamma*np.power(h_func, power).reshape((1,1)) )

        return h_func


    def add_velocity_bound(self, vel_limit):
        scale = 1 / vel_limit
        G = np.vstack((np.eye(3), -np.eye(3))) * scale
        h = np.ones([6, 1]) * vel_limit * scale
        self.__set_constraint( G, h )

    # TODO: add area with boundary