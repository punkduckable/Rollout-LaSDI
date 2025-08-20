# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add the main directory to the search path.
import  os;
import  sys;
src_Path        : str   = os.path.dirname(os.path.dirname(__file__));
util_Path       : str   = os.path.join(src_Path, "Utilities");
sys.path.append(src_Path);
sys.path.append(util_Path);

import  logging;

import  numpy;
import  torch;

from    LatentDynamics      import  LatentDynamics;
from    FiniteDifference    import  Derivative1_Order4, Derivative1_Order2_NonUniform;
from    FirstOrderSolvers   import  RK4;

# Setup logger.
LOGGER  : logging.Logger    = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# SINDy class
# -------------------------------------------------------------------------------------------------

class SINDy(LatentDynamics):
    def __init__(   self, 
                    n_z             : int,
                    coef_norm_order : str | float,
                    Uniform_t_Grid  : bool) -> None:
        r"""
        Initializes a SINDy object. This is a subclass of the LatentDynamics class which uses the 
        SINDy algorithm as its model for the ODE governing the latent state. Specifically, we 
        assume there is a library of functions, f_1(z), ... , f_N(z), each one of which is a 
        monomial of the components of the latent space, z, and a set of coefficients c_{i,j}, 
        i = 1, 2, ... , n_z and j = 1, 2, ... , N such that

            z_i'(t) = \sum_{j = 1}^{N} c_{i,j} f_j(z)
        
        In this case, we assume that f_1, ... , f_N consists of the set of order <= 1 monomials. 
        That is, f_1(z), ... , f_N(z) = 1, z_1, ... , z_{n_z}.
            

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        n_z : int
            The number of dimensions in the latent space, where the latent dynamics takes place.

        coef_norm_order : float, 'inf', 'fro'
            Specifies which norm we want to use when computing the coefficient loss. We pass this 
            as the "p" argument to torch.norm. If it's a float, coef_norm_order = p \in \mathbb{R}, 
            then we use the corresponding l^p norm. If it is "inf" or "fro", we use the infinity 
            or Frobenius norm, respectively. 

        Uniform_t_Grid : bool 
            If True, then for each parameter value, the times corresponding to the frames of the 
            solution for that parameter value will be uniformly spaced. In other words, the first 
            frame corresponds to time t0, the second to t0 + h, the k'th to t0 + (k - 1)h, etc 
            (note that h may depend on the parameter value, but it needs to be constant for a 
            specific parameter value). The value of this setting determines which finite difference 
            method we use to compute time derivatives. 

            
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        Nothing!
        """

        # Run the base class initializer. The only thing this does is set the n_z and n_t 
        # attributes.
        super().__init__(n_z                = n_z, 
                         coef_norm_order    = coef_norm_order, 
                         Uniform_t_Grid     = Uniform_t_Grid);
        LOGGER.info("Initializing a SINDY object with n_z = %d, coef_norm_order = %s, Uniform_t_Grid = %s" % (  self.n_z, 
                                                                                                                str(self.coef_norm_order), 
                                                                                                                str(self.Uniform_t_Grid)));

        # Set n_IC and n_coefs.
        # We only allow library terms of order <= 1. If we let z(t) \in \mathbb{R}^{n_z} denote the 
        # latent state at some time, t, then the possible library terms are 1, z_1(t), ... , 
        # z_{n_z}(t). Since each component function gets its own set of coefficients, there must 
        # be n_z*(n_z + 1) total coefficients.
        #TODO(kevin): generalize for high-order dynamics
        self.n_coefs    : int   = self.n_z*(self.n_z + 1);
        self.n_IC       : int   = 1;

        # TODO(kevin): other loss functions
        self.MSE = torch.nn.MSELoss(reduction = 'mean');
        self.MAE = torch.nn.L1Loss(reduction = 'mean');
        return;
    


    def calibrate(  self,  
                    Latent_States   : list[list[torch.Tensor]], 
                    loss_type       : str,
                    t_Grid          : list[torch.Tensor], 
                    input_coefs     : list[torch.Tensor] = []) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        This function computes the optimal SINDy coefficients using the current latent time 
        series. Specifically, let us consider the case when Z has two dimensions (the case when 
        it has three is identical, just with different coefficients for each instance of the 
        leading dimension of Z). In this case, we assume that the rows of Z correspond to a 
        trajectory of latent states. Specifically, we assume the i'th row holds the latent state,
        z, at time t_0 + i*dt. We use SINDy to find the coefficients in the dynamical system
        z'(t) = C \Phi(z(t)), where C is a matrix of coefficients and \Phi(z(t)) represents a
        library of terms. We find the matrix C corresponding to the dynamical system that best 
        agrees with the data in the rows of Z. 


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th list element is a one element list whose only element is a 2d numpy array of 
            shape (n_t(i), n_z) whose p, q element holds the q'th component of the latent state 
            during the p'th time step (whose time value corresponds to the p'th element of t_Grid) 
            when we use the i'th combination of parameter values. 
        
        loss_type : str
            The type of loss function to use. Must be either "MSE" or "MAE".

        t_Grid : list[torch.Tensor], len = n_param
            i'th element should be a 1d tensor of shape (n_t(i)) whose j'th element holds the time 
            value corresponding to the j'th frame when we use the i'th combination of parameter 
            values.

        input_coefs : list[torch.Tensor], len = n_param, optional
            The i'th element of this list is a 1d tensor of shape (n_coefs) holding the 
            coefficients for the i'th combination of parameter values. If input_coefs is None, 
            then we will learn the coefficients using Least Squares. If input_coefs is not None, 
            then we will use the provided coefficients to compute the loss.

            
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        output_coefs, loss_sindy, loss_coef. 
        
        output_coefs : torch.Tensor, shape = (n_train, n_coef)
            A matrix of shape (n_train, n_coef), where n_train is the number of parameter 
            combinations in the training set and n_coef is the number of coefficients in the latent 
            dynamics. The i,j entry of this array holds the value of the j'th coefficient when we 
            use the i'th combination of parameter values.

        loss_sindy : torch.Tensor, shape = []
            A 0-dimensional tensor whose lone element holds the sum of the SINDy losses across the 
            set of combinations of parameters in the training set. 

        loss_coef : torch.Tensor, shape = [] 
            A 0-dimensional tensor whose lone element holds the sum of the L1 norms of the 
            coefficients across the set of combinations of parameters in the training set.
        """

        # Run checks.
        assert(isinstance(t_Grid, list));
        assert(isinstance(Latent_States, list));
        assert(len(Latent_States)   == len(t_Grid));

        n_param : int   = len(t_Grid);
        n_IC    : int   = 1;
        n_z     : int   = self.n_z;
        for i in range(n_param):
            assert(isinstance(Latent_States[i], list));
            assert(len(Latent_States[i]) == n_IC);

            for j in range(n_IC):
                assert(isinstance(Latent_States[i][j], torch.Tensor));
                assert(len(Latent_States[i][j].shape)   == 2);
                assert(Latent_States[i][j].shape[-1]    == n_z);

        # Run checks on loss_type.
        assert(loss_type in ["MSE", "MAE"]);

        # Run checks on input_coefs.
        assert(isinstance(input_coefs, list));
        if(len(input_coefs) > 0):
            assert(isinstance(input_coefs, list));
            assert(len(input_coefs) == n_param);
            for i in range(n_param):
                assert(isinstance(input_coefs[i], torch.Tensor));
                assert(len(input_coefs[i].shape) == 1);
                assert(input_coefs[i].shape[0] == self.n_coefs);


        # -----------------------------------------------------------------------------------------
        # If there are multiple combinations of parameter values, loop through them.
        
        if (n_param > 1):
            # Prepare an array to house the flattened coefficient matrices for each combination of
            # parameter values.
            output_coefs_list : list[torch.Tensor] = [];

            # Compute the losses, coefficients for each combination of parameter values.
            loss_sindy  = torch.zeros(1, dtype = torch.float32);
            loss_coef   = torch.zeros(1, dtype = torch.float32);
            for i in range(n_param):
                """"
                Get the optimal SINDy coefficients for the i'th combination of parameter values. 
                Remember that Latent_States[i][0] is a tensor of shape (n_t(j), n_z) whose (j, k) 
                entry holds the k'th component of the j'th frame of the latent trajectory for the 
                i'th combination of parameter values. 
                
                Note that Result a 3 element tuple.
                """
                if(len(input_coefs) == 0):
                    result : tuple[torch.Tensor, torch.Tensor, torch.Tensor]    = self.calibrate(   Latent_States = [Latent_States[i]], 
                                                                                                    t_Grid        = [t_Grid[i]],
                                                                                                    loss_type     = loss_type);
                else:
                    result                      = self.calibrate(Latent_States  = [Latent_States[i]], 
                                                                  t_Grid        = [t_Grid[i]],
                                                                  input_coefs   = [input_coefs[i]],
                                                                  loss_type     = loss_type);

                # Package the results from this combination of parameter values.
                output_coefs_list.append(result[0]);
                loss_sindy  = loss_sindy + result[1];
                loss_coef   = loss_coef + result[2];
            
            # Package everything to return!
            return torch.stack(output_coefs_list), loss_sindy, loss_coef;
            

        # -----------------------------------------------------------------------------------------
        # Evaluate for one combination of parameter values case.

        t_Grid0 : torch.Tensor  = t_Grid[0];
        Z       : torch.Tensor  = Latent_States[0][0];
        n_t     : int           = len(t_Grid0);

        # First, compute the time derivatives. Which method we use depends on if we have a uniform 
        # grid spacing or not. If so, we use an O(h^4) method. Otherwise, we use an O(h^2) one. In
        # either case, this yields a 2d torch.Tensor object of shape (n_t, n_z) whose i,j element 
        # holds the holds an approximation of (d/dt) Z_j(t_Grid0[i]).
        if(self.Uniform_t_Grid == True):
            h       : float         = (t_Grid0[1] - t_Grid0[0]).item();
            dZdt    : torch.Tensor  = Derivative1_Order4(Z, h);
        else:
            dZdt                    = Derivative1_Order2_NonUniform(Z, t_Grid = t_Grid0);

        # Concatenate a column of ones. This will correspond to a constant term in the latent 
        # dynamics.
        Z_1     : torch.Tensor  = torch.cat([torch.ones(n_t, 1), Z], dim = 1)
        
        if(len(input_coefs) == 0):
            # For each j, solve the least squares problem 
            #   min{ || dZdt[:, j] - Z_1 c_j|| : C_j \in \mathbb{R}ˆNl }
            # where Nl is the number of library terms (in this case, just n_z + 1, since we only allow
            # constant and linear terms). We store the resulting solutions in a matrix, coefs, whose 
            # j'th column holds the results for the j'th column of dZdt. Thus, coefs is a 2d tensor
            # with shape (Nl, n_z).
            coefs                   = torch.linalg.lstsq(Z_1, dZdt).solution
        else:
            coefs   : torch.Tensor  = input_coefs[0].reshape(self.n_z + 1, self.n_z);

        # Compute the losses.
        if(loss_type == "MSE"):
            loss_sindy = self.MSE(dZdt, Z_1 @ coefs);
        elif(loss_type == "MAE"):
            loss_sindy = self.MAE(dZdt, Z_1 @ coefs);
        # NOTE(kevin): by default, this will be L1 norm.
        loss_coef = torch.norm(coefs, self.coef_norm_order)

        # Prepare coefs and the losses to return. Note that we flatten the coefficient matrix.
        # Note: output of lstsq is not contiguous in memory.
        output_coefs   : torch.Tensor  = coefs.flatten()
        return output_coefs, loss_sindy, loss_coef



    def simulate(   self,
                    coefs   : numpy.ndarray           | torch.Tensor, 
                    IC      : list[list[numpy.ndarray | torch.Tensor]],
                    t_Grid  : list[numpy.ndarray      | torch.Tensor]) -> list[list[numpy.ndarray | torch.Tensor]]:
        """
        Time integrates the latent dynamics from multiple initial conditions for each combination
        of coefficients in coefs. 
 

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        coefs : numpy.ndarray or torch.Tensor, shape = (n_param, n_coef)
            i'th row represents the optimal set of coefficients when we use the i'th combination 
            of parameter values. We inductively call simulate on each row of coefs. 

        IC : list[list[numpy.ndarray]] or list[list[torch.Tensor]], len = n_param
            i'th element is an n_IC element list whose j'th element is a 2d numpy.ndarray or 
            torch.Tensor object of shape (n(i), n_z). Here, n(i) is the number of initial 
            conditions (for a fixed set of coefficients) we want to simulate forward using the i'th 
            set of coefficients. Further, n_z is the latent dimension. If you want to simulate a 
            single IC, for the i'th set of coefficients, then n(i) == 1. IC[i][j][k, :] should hold 
            the k'th initial condition for the j'th derivative of the latent state when we use the 
            i'th combination of parameter values. 

        t_Grid : list[numpy.ndarray] or list[torch.Tensor], len = n_param
            i'th entry is a 2d numpy.ndarray or torch.Tensor whose shape is either (n(i), n_t(i)) 
            or shape (n_t(i)). The shape should be 2d if we want to use different times for each 
            initial condition and 1d if we want to use the same times for all initial conditions. 
        
            In the former case, the j,k array entry specifies k'th time value at which we solve for 
            the latent state when we use the j'th initial condition and the i'th set of 
            coefficients. Each row should be in ascending order. 
        
            In the latter case, the j'th entry should specify the j'th time value at which we solve 
            for each latent state when we use the i'th combination of parameter values.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------        
        
        Z : list[list[numpy.ndarray]] or list[list[torch.Tensor]], len = n_parm
            i'th element is a list of length n_IC whose j'th entry is a 3d array of shape 
            (n_t(i), n(i), n_z). The p, q, r entry of this array should hold the r'th component of 
            the p'th frame of the j'th tine derivative of the solution to the latent dynamics when 
            we use the q'th initial condition for the i'th combination of parameter values.
        """

        # Run checks.
        assert(len(coefs.shape)     == 2);
        n_param : int = coefs.shape[0];
        assert(isinstance(t_Grid, list));
        assert(isinstance(IC, list));
        assert(len(IC)              == n_param);
        assert(len(t_Grid)          == n_param);
        
        assert(isinstance(IC[0], list));
        n_IC : int = len(IC[0]);
        assert(n_IC == 1);
        for i in range(n_param):
            assert isinstance(IC[i], list),                                     "IC[%d] is not a list" % i;
            assert len(IC[i]) == n_IC,                                          "len(IC[%d]) = %d, n_IC = %d" % (i, len(IC[i]), n_IC);
            assert len(t_Grid[i].shape) == 2 or len(t_Grid[i].shape) == 1,      "len(t_Grid[%d].shape) = %d" % (i, len(t_Grid[i].shape));
            for j in range(n_IC):
                assert len(IC[i][j].shape) == 2,                                "IC[%d][%d].shape = %s" % (i, j, str(IC[i][j].shape));
                assert type(coefs)          == type(IC[i][j]),                  "type(coefs) = %s, type(IC[%d][%d]) = %s" % (str(type(coefs)), i, j, str(type(IC[i][j])));
                assert IC[i][j].shape[1]    == self.n_z,                        "IC[%d][%d].shape[1] = %d, self.n_z = %d" % (i, j, IC[i][j].shape[1], self.n_z);
                if(len(t_Grid[i].shape) == 2):
                    assert t_Grid[i].shape[0] == IC[i][j].shape[0];


        # -----------------------------------------------------------------------------------------
        # If there are multiple combinations of coefficients, loop through them.
        
        if(n_param > 1):
            LOGGER.debug("Simulating with %d parameter combinations" % n_param);

            # Cycle through the parameter combinations
            Z   : list[list[numpy.ndarray | torch.Tensor]]  = [];
            for i in range(n_param): 
                # Fetch the i'th set of coefficients, the corresponding collection of initial
                # conditions, and the set of time values.
                ith_coefs   : numpy.ndarray             | torch.Tensor              = coefs[i, :].reshape(1, -1);
                ith_IC      : list[list[numpy.ndarray   | torch.Tensor]]            = [IC[i]];
                ith_t_Grid  : list[numpy.ndarray | torch.Tensor]                    = [t_Grid[i]];

                # Call this function using them. This should return a 1 element holding the 
                # the solution for the i'th combination of parameter values.
                ith_Results : list[numpy.ndarray | torch.Tensor]    = self.simulate(coefs   = ith_coefs, 
                                                                                    IC      = ith_IC, 
                                                                                    t_Grid  = ith_t_Grid)[0];

                # Add these results to Z.
                Z.append(ith_Results);

            # All done.
            return Z;
        

        # -----------------------------------------------------------------------------------------
        # Evaluate for one combination of parameter values case.

        # In this case, there is just one parameter. Extract t_Grid, which has shape 
        # (n(i), n_t(i)) or (n_t(i)).
        t_Grid0  : numpy.ndarray | torch.Tensor  = t_Grid[0];
        if(isinstance(t_Grid0, torch.Tensor)):
            t_Grid0 = t_Grid0.detach().numpy();
        n_t_i   : int           = t_Grid0.shape[-1];
        if(len(t_Grid0.shape) == 1):
            Same_t_Grid : bool  = True;
        else:
            Same_t_Grid         = False;

        # coefs has shape (1, n_coefs). Each element of IC should have shape (n(i), n_z). 
        Z0  : numpy.ndarray | torch.Tensor  = IC[0][0]; 
        n_i : int                           = Z0.shape[0];

        # First, we need to extract the matrix of coefficients. We know that coefs is the least 
        # squares solution to dZ_dt = hstack[1, Z] E^T. 
        E   : numpy.ndarray | torch.Tensor = coefs.reshape([self.n_z + 1, self.n_z]).T;

        # Extract A and b. Note that we need to reshape b to have shape (1, n_z) to enable
        # broadcasting.
        b   : numpy.ndarray | torch.Tensor = E[:, 0 ].reshape(1, -1);
        A   : numpy.ndarray | torch.Tensor = E[:, 1:];


        # Set up a lambda function to approximate 
        #   z'(t) \approx b + A z(t)
        # In this case, we expect dz_dt and z to have shape (n(i), n_z). Thus, matmul(z, A.T) will 
        # have shape (n(i), n_z). The i'th row of this should hold the z portion of the rhs of the 
        # latent dynamics for the i'th IC. Similar results hold for dot(dz_dt, C.T). The final 
        # result should have shape (n, n_z). The i'th row should hold the rhs of the latent 
        # dynamics for the i'th IC.
        if(isinstance(coefs, numpy.ndarray)):
            f   = lambda t, z: b + numpy.matmul(z, A.T);
        if(isinstance(coefs, torch.Tensor)):
            f   = lambda t, z: b + torch.matmul(z, A.T);

        # Solve the ODE forward in time. U should have shape (n_t, n(i), n_z). If we use the 
        # same t values for each IC, then we can exploit the fact that the latent dynamics are 
        # autonomous to solve using each IC simultaneously. Otherwise, we need to run the latent
        # dynamics one IC at a time. 
        if(Same_t_Grid == True):
            Z = [[RK4(f = f, y0 = Z0, t_Grid = t_Grid0)]]; 
        else:
            # Cycle through the ICs.
            Z_list : list[torch.Tensor | numpy.ndarray] = [];   
            for j in range(n_i):
                Z_j         = RK4(f = f, y0 = Z0[j, :].reshape(1, -1), t_Grid = t_Grid0[j, :]);
                Z_list.append(Z_j);

            # Stack the results.
            if(isinstance(coefs, numpy.ndarray)):
                Z = [[numpy.concatenate(Z_list, axis = 1)]];    # shape = (n_t, n_i, n_z)
            elif(isinstance(coefs, torch.Tensor)):
                Z = [[torch.cat(Z_list, dim = 1)]];            # shape = (n_t, n_i, n_z)
        
        # All done!
        return Z;