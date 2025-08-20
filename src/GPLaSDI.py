# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  sys;
import  os;
Physics_Path    : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Physics"));
LD_Path         : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "LatentDynamics"));
Utils_Path      : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Utilities"));
sys.path.append(Physics_Path);
sys.path.append(LD_Path);
sys.path.append(Utils_Path);

import  logging;

import  torch;
import  numpy;
from    torch.optim                 import  Optimizer;
from    sklearn.gaussian_process    import  GaussianProcessRegressor;
from    scipy                       import  interpolate;

from    GaussianProcess             import  sample_coefs, fit_gps;
from    Model                       import  Autoencoder;
from    Timing                      import  Timer;
from    ParameterSpace              import  ParameterSpace;
from    Physics                     import  Physics;
from    LatentDynamics              import  LatentDynamics;
from    SolveROMs                   import  get_FOM_max_std;
from    FiniteDifference            import  Derivative1_Order4, Derivative1_Order2_NonUniform;
from    Logging                     import  Log_Dictionary;

# Setup Logger
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# BayesianGLaSDI class
# -------------------------------------------------------------------------------------------------

# move optimizer parameters to device
def optimizer_to(optim : Optimizer, device : str) -> None:
    """
    This function moves an optimizer object to a specific device. 


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    optim : Optimizer
        The optimizer whose device we want to change.

    device : str
        The device we want to move optim onto. 


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Nothing.
    """

    # Cycle through the optimizer's parameters.
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device);
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device);
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device);
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device);



class BayesianGLaSDI:
    # An n_Train element list. The i'th element is is an n_IC element list whose j'th element is a
    # numpy ndarray of shape (n_t(i), Frame_Shape) holding a sequence of samples of the j'th 
    # derivative of the FOM solution when we use the i'th combination of training values. 
    U_Train : list[list[torch.Tensor]]  = [];  

    # An n_Train element list whose i'th element is a torch.Tensor of shape (n_t(i)) whose j'th
    # element holds the time value for the j'th frame when we use the i'th combination of training 
    # parameters.
    t_Train : list[torch.Tensor]        = []; 

    # An n_Train element list whose i'th element is an n_IC element list whose j'th element is a
    # float holding the std of the j'th derivative of the FOM solution when we use the i'th 
    # combination of training parameters.
    std_Train : list[list[float]]       = [];

    # Same as U_Test, but used for the test set.
    U_Test  : list[list[torch.Tensor]]  = [];  

    # An n_Test element list whose i'th element is a torch.Tensor of shape (n_t(i)) whose j'th
    # element holds the time value for the j'th frame when we use the i'th combination of testing 
    # parameters.
    t_Test  : list[torch.Tensor]        = [];

    # number of IC's in the FOM solution.
    n_IC  : int;



    def __init__(self, 
                 physics            : Physics, 
                 model              : torch.nn.Module, 
                 latent_dynamics    : LatentDynamics, 
                 param_space        : ParameterSpace, 
                 config             : dict):
        """
        This class runs a full GPLaSDI training. As input, it takes the model defined as a 
        torch.nn.Module object, a Physics object to recover FOM ICs + information on the time 
        discretization, a latent dynamics object, and a parameter space object (which holds the 
        testing and training sets of parameters).

        The "train" method runs the active learning training loop, computes the reconstruction and 
        SINDy loss, trains the GPs, and samples a new FOM data point.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        physics : Physics
            Encodes the FOM model. It allows us to fetch the FOM solution and/or initial conditions 
            for a particular combination of parameters. We use this object to generate FOM 
            solutions which we then use to train the model and latent dynamics.
         
        model : torch.nn.Module
            use to compress the FOM state to a reduced, latent state.

        latent_dynamics : LatentDynamics
            A LatentDynamics object which describes how we specify the dynamics in the model's 
            latent space.

        param_space: ParameterSpace
            holds the set of testing and training parameters. 

        config: dict
            houses the LaSDI settings. This should be the 'lasdi' sub-dictionary of the config 
            file.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """
        
        # Checks.
        n_IC    : int           = latent_dynamics.n_IC;
        assert model.n_IC       == n_IC, "model.n_IC = %d, n_IC = %d" % (model.n_IC, n_IC);
        assert physics.n_IC     == n_IC, "physics.n_IC = %d, n_IC = %d" % (physics.n_IC, n_IC);
        self.n_IC               = n_IC;

        LOGGER.info("Initializing a GPLaSDI object"); 
        Log_Dictionary(LOGGER = LOGGER, D = config, level = logging.INFO);

        self.physics                        = physics;
        self.model                          = model;
        self.latent_dynamics                = latent_dynamics;
        self.param_space                    = param_space;
        
        # Initialize a timer object. We will use this while training.
        self.timer                          = Timer();

        # Extract training/loss hyperparameters from the configuration file. 
        self.lr                     : float     = config['lr'];                     # Learning rate for the optimizer.
        self.n_samples              : int       = config['n_samples'];              # Number of samples to draw per coefficient per combination of parameters
        self.p_rollout_init         : float     = config['p_rollout_init'];         # The proportion of the simulated we simulate forward when computing the rollout loss.
        self.rollout_update_freq    : float     = config['rollout_update_freq'];    # We increase p_rollout after this many iterations.
        self.dp_per_update          : float     = config['dp_per_update'];          # We increase p_rollout by this much each time we increase it.
        self.randomized_rollout     : bool      = config['randomized_rollout'];     # If true, then the duration is randomized for each IC, with p_rollout acting as a maximum. Otherwise, everything is rolled out by p_rollout.
        self.rollout_spline_order   : int       = config['rollout_spline_order'];   # The order of the spline used to interpolate the rollout targets.
        self.p_IC_rollout_init      : float     = config['p_IC_rollout_init'];      # The proportion of the simulation we simulate forward when computing the IC rollout loss.
        self.IC_rollout_update_freq : float     = config['IC_rollout_update_freq']; # We increase p_IC_rollout after this many iterations.
        self.IC_dp_per_update       : float     = config['IC_dp_per_update'];       # We increase p_IC_rollout by this much each time we increase it.
        self.n_iter                 : int       = config['n_iter'];                 # Number of iterations for one train and greedy sampling
        self.max_iter               : int       = config['max_iter'];               # We stop training if restart_iter goes above this number. 
        self.max_greedy_iter        : int       = config['max_greedy_iter'];        # We stop performing greedy sampling if restart_iter goes above this number.
        self.loss_weights           : dict      = config['loss_weights'];           # A dictionary housing the weights of the various parts of the loss function.
        self.loss_types             : dict      = config['loss_types'];             # A dictionary housing the type of loss function (MSE or MAE) for each part of the loss function.
        self.learnable_coefs        : bool      = config['learnable_coefs'];        # If True, the latent dynamics coefficients are learnable parameters. If false, we compute them using Least Squares.

        # Set the device to train on. We default to cpu.
        device = config['device'] if 'device' in config else 'cpu';
        if (device == 'cuda'):
            assert(torch.cuda.is_available());
            self.device = device;
        elif (device == 'mps'):
            assert(torch.backends.mps.is_available());
            self.device = device;
        else:
            self.device = 'cpu';

        # If we are learning the latent dynamics coefficients, then we need to set up 
        # a torch Parameter housing the the coefficients for each combination of testing 
        # parameters. If we aren't learning the coefficients, then this will never be used.
        self.test_coefs : torch.Tensor = torch.nn.parameter.Parameter(torch.zeros(self.param_space.n_test(), self.latent_dynamics.n_coefs, dtype = torch.float32, device = self.device, requires_grad = True));

        # Set up the optimizer and loss function.
        LOGGER.info("Setting up the optimizer with a learning rate of %f" % (self.lr));
        self.optimizer          : Optimizer = torch.optim.Adam(list(model.parameters()) + [self.test_coefs], lr = self.lr);
        self.MSE                            = torch.nn.MSELoss(reduction = 'mean');
        self.MAE                            = torch.nn.L1Loss(reduction = 'mean');

        # Set paths for checkpointing. 
        self.path_checkpoint    : str       = os.path.join(os.path.pardir, "checkpoint");
        self.path_results       : str       = os.path.join(os.path.pardir, "results");

        # Make sure the checkpoints and results directories exist.
        from pathlib import Path;
        Path(os.path.dirname(self.path_checkpoint)).mkdir(  parents = True, exist_ok = True);
        Path(os.path.dirname(self.path_results)).mkdir(     parents = True, exist_ok = True);

        # Set up variables to aide checkpointing
        self.best_coefs     = numpy.zeros((self.param_space.n_test(), self.latent_dynamics.n_coefs), dtype = numpy.float32);
        self.best_epoch     = -1;
        self.restart_iter   = 0;                # Iteration number at the end of the last training period
        
        # All done!
        return;



    def train(self, reset_optim : bool = True) -> None:
        """
        Runs a round of training on the model.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        reset_optim : bool
            If True, we re-initialize self's optimizer before training. 



        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """
        
        # Make sure we have at least one training data point.
        assert len(self.U_Train) > 0, "len(self.U_Train) = %d" % len(self.U_Train);
        assert len(self.U_Train) == self.param_space.n_train(), "len(self.U_Train) = %d, self.param_space.n_train() = %d" % (len(self.U_Train), self.param_space.n_train());


        # Reset optimizer, if desirable. 
        if(reset_optim == True): self._reset_optimizer();


        # -------------------------------------------------------------------------------------
        # Setup. 

        # Fetch parameters. Note that p_rollout and p_IC_rollout can be negative.
        n_train             : int               = self.param_space.n_train();
        p_rollout           : float             = min(0.75, self.p_rollout_init + self.dp_per_update*(self.restart_iter//self.rollout_update_freq));
        p_IC_rollout        : float             = min(1.0, self.p_IC_rollout_init + self.IC_dp_per_update*(self.restart_iter//self.IC_rollout_update_freq));
        LD                  : LatentDynamics    = self.latent_dynamics;
        best_loss           : float             = numpy.inf;                    # Stores the lowest loss we get in this round of training.

        # Map everything to self's device.
        device              : str                       = self.device;
        model_device        : torch.nn.Module           = self.model.to(device);
        U_Train_device      : list[list[torch.Tensor]]  = [];
        t_Train_device      : list[torch.Tensor]        = [];
        for i in range(n_train):
            t_Train_device.append(self.t_Train[i].to(device));
            
            ith_U_Train_device  : list[torch.Tensor] = [];
            for j in range(self.n_IC):
                ith_U_Train_device.append(self.U_Train[i][j].to(device));
            U_Train_device.append(ith_U_Train_device);

        # Make sure the checkpoints and results directories exist.
        from pathlib import Path
        Path(self.path_checkpoint).mkdir(   parents = True, exist_ok = True);
        Path(self.path_results).mkdir(      parents = True, exist_ok = True);

        # Rollout setup
        if(self.loss_weights['rollout'] > 0 and p_rollout > 0):
            self.timer.start("Rollout Setup");

            t_Grid_rollout, n_rollout_ICs, U_Target_Rollout_Trajectory = self._rollout_setup(
                                                                            t            = t_Train_device, 
                                                                            U            = U_Train_device, 
                                                                            p_rollout    = p_rollout);
            self.timer.end("Rollout Setup");

        # IC rollout setup
        if(self.loss_weights['IC_rollout'] > 0 and p_IC_rollout > 0):
            self.timer.start("IC Rollout Setup");

            t_Grid_IC_rollout, n_IC_rollout_frames, U_IC_Rollout_Targets = self._IC_rollout_setup(  t            = t_Train_device, 
                                                                                                    p_IC_rollout = p_IC_rollout);
            self.timer.end("IC Rollout Setup"); 

        # If we are learning the latent dynamics coefficients, then we need to determine 
        # which combinations of parameters are in the training set. Specifically, each 
        # element of the train space should also be in the test space. We need to figure out 
        # the index of each train space element within the test space.
        train_coefs_list : list[torch.Tensor] = [];
        if(self.learnable_coefs == True):
            train_coefs_list : list[torch.Tensor] = [];
            for i in range(n_train):
                ith_train_in_test : bool = False;
                for j in range(self.param_space.n_test()):
                    if(numpy.all(self.param_space.test_space[j, :] == self.param_space.train_space[i, :])):
                        train_coefs_list.append(self.test_coefs[j, :]);
                        ith_train_in_test = True;
                        break;

                # Make sure we found the training combination of parameters in the test space.
                assert(ith_train_in_test == True);


        # -----------------------------------------------------------------------------------------
        # Run the iterations!

        next_iter   : int = min(self.restart_iter + self.n_iter, self.max_iter);
        LOGGER.info("Training for %d epochs (starting at %d, going to %d) with %d parameters" % (next_iter - self.restart_iter, self.restart_iter, next_iter, n_train));
        
        for iter in range(self.restart_iter, next_iter):
            self.timer.start("train_step");

            # Check if we need to update p_rollout. If so, then we also need to update 
            # t_Grid_rollout, n_rollout_ICs, and U_Target_Rollout_Trajectory
            if(self.loss_weights['rollout'] > 0 and iter > 0 and ((iter % self.rollout_update_freq) == 0)):
                self.timer.start("Rollout Setup");

                p_rollout  += self.dp_per_update;
                p_rollout   = min(0.75, p_rollout);

                LOGGER.info("p_rollout is now %f (increased %f)" % (p_rollout, self.dp_per_update));

                if(p_rollout > 0):
                    t_Grid_rollout, n_rollout_ICs, U_Target_Rollout_Trajectory = self._rollout_setup(
                                                                                    t            = t_Train_device, 
                                                                                    U            = U_Train_device, 
                                                                                    p_rollout    = p_rollout);
                
                self.timer.end("Rollout Setup");

            # Check if we need to update IC rollout parameters
            if(self.loss_weights['IC_rollout'] > 0 and iter > 0 and ((iter % self.IC_rollout_update_freq) == 0)):
                self.timer.start("IC Rollout Setup");

                p_IC_rollout  += self.IC_dp_per_update;
                p_IC_rollout   = min(1.0, p_IC_rollout);

                LOGGER.info("p_IC_rollout is now %f (increased %f)" % (p_IC_rollout, self.IC_dp_per_update));

                # Setup IC rollout time grids and targets
                if(p_IC_rollout > 0):
                    t_Grid_IC_rollout, n_IC_rollout_frames, U_IC_Rollout_Targets = self._IC_rollout_setup(  t            = t_Train_device, 
                                                                                                            p_IC_rollout = p_IC_rollout);
                
                self.timer.end("IC Rollout Setup"); 

            self.optimizer.zero_grad();


            # -------------------------------------------------------------------------------------
            # Compute losses

            if(isinstance(model_device, Autoencoder)):
                # Setup. 
                Latent_States       : list[list[torch.Tensor]]  = [];       # len = n_train. i'th element is 1 element list of (n_t_i, n_z) arrays.

                loss_recon          : torch.Tensor              = torch.zeros(1, dtype = torch.float32, device = device);
    
                ROM_Rollout_ICs     : list[list[torch.Tensor]]  = [];       # len = n_train. i'th element is 1 element list of (n_rollout_ICs[i], n_z) arrays.
                FOM_Rollout_Targets : list[list[torch.Tensor]]  = [];       # len = n_train. i'th element is 1 element list of (n_rollout_ICs[i], physics.Frame_Shape) arrays holding the FOM rollout targets.
                ROM_Rollout_Targets : list[list[torch.Tensor]]  = [];       # len = n_train. i'th element is 1 element list of (n_rollout_ICs[i], n_z) arrays holding the ROM rollout targets.
                Rollout_Indices     : list[int]                 = [];       # len = n_train. i'th element is an array of shape (n_rollout_ICs[i]) specifying the indices (in rollout trajectories) of the frames we use as rollout targets.

                # Cycle through the combinations of parameter values
                for i in range(n_train):
                    # Setup. 
                    U_i         : torch.Tensor  = U_Train_device[i][0];
                    t_Grid_i    : torch.Tensor  = t_Train_device[i];
                    n_t_i       : int           = t_Grid_i.shape[0];


                    # -----------------------------------------------------------------------------
                    # Forward pass

                    self.timer.start("Forward Pass");

                    # Run the forward pass. This results in an n_train element list whose i'th 
                    # element is a 1 element list whose only element is a tensor of shape 
                    # (n_t(i), physics.Frame_Shape) whose [k, ...] slice holds our prediction for 
                    # the FOM solution at time t_Grid[i][k] when we use the i'th combination of 
                    # parameter values. Here, n_t(i) is the number of time steps in the solution 
                    # for the i'th combination of parameter values. 
                    Z_i         : torch.Tensor  = model_device.Encode(U_i);
                    Latent_States.append([Z_i]);
                    U_Pred_i    : torch.Tensor  = model_device.Decode(Z_i);

                    self.timer.end("Forward Pass");


                    # ----------------------------------------------------------------------------
                    # Reconstruction loss

                    if(self.loss_weights['recon'] > 0):
                        self.timer.start("Reconstruction Loss");

                        if(self.loss_types['recon'] == "MSE"):
                            loss_recon += self.MSE(U_i, U_Pred_i) / (self.std_Train[i][0]**2);
                        elif(self.loss_types['recon'] == "MAE"):
                            loss_recon += self.MAE(U_i, U_Pred_i) / self.std_Train[i][0];
                        
                        self.timer.end("Reconstruction Loss");


                    # ----------------------------------------------------------------------------
                    # Setup Rollout losses.

                    if(self.loss_weights['rollout'] > 0 and p_rollout > 0):
                        self.timer.start("Rollout Setup");

                        # Select the latent states we want to use as initial conditions for the i'th 
                        # combination of parameter values. This should be the first 
                        # n_rollout_ICs[i] frames (n_rollout_ICs[i] is computed such that if we 
                        # simulate the first n_rollout_ICs[i] frames, the final times are less than 
                        # the final time for this combination of parameter values. Each element of 
                        # ROM_Rollout_IC is a 1 element list of torch.Tensor objects of shape 
                        # (n_rollout_ICs[i], n_z).
                        ROM_Rollout_ICs.append([Z_i[:n_rollout_ICs[i], :]]);

                        if(self.randomized_rollout == True):
                            # Generate the indices of the frames we want to use as the targets.
                            Rollout_Indices_i   : numpy.ndarray = numpy.random.randint(0, t_Grid_rollout[i].shape[0], n_rollout_ICs[i]);
                            Rollout_Indices.append(Rollout_Indices_i);

                            # Fetch the corresponding targets.
                            FOM_Rollout_Targets_i : list[torch.Tensor] = [];
                            for j in range(self.n_IC):
                                FOM_Rollout_Targets_ij : numpy.ndarray = numpy.empty((n_rollout_ICs[i],) + tuple(self.physics.Frame_Shape), dtype = numpy.float32);
                                
                                # Fetch the target solution at the target time for each IC.
                                for k in range(n_rollout_ICs[i]):
                                    FOM_Rollout_Targets_ij[k, ...] = U_Target_Rollout_Trajectory[i][j][k, Rollout_Indices_i[k], ...];
                                FOM_Rollout_Targets_i.append(torch.tensor(FOM_Rollout_Targets_ij, dtype = torch.float32, device = device));
                            FOM_Rollout_Targets.append(FOM_Rollout_Targets_i);
                    
                        else:
                            # In this case, the targets are the final frames of the rolled out 
                            # trajectories.
                            FOM_Rollout_Targets_i : list[torch.Tensor] = [];
                            for j in range(self.n_IC):
                                FOM_Rollout_Targets_ij : numpy.ndarray = U_Target_Rollout_Trajectory[i][j][:, -1, ...];
                                FOM_Rollout_Targets_i.append(torch.tensor(FOM_Rollout_Targets_ij, dtype = torch.float32, device = device));
                            FOM_Rollout_Targets.append(FOM_Rollout_Targets_i);


                        # Fetch the corresponding target by encoding the FOM targets using the 
                        # current encoder.
                        ROM_Rollout_Targets.append([model_device.Encode(*FOM_Rollout_Targets_i)]);

                        self.timer.end("Rollout Setup");


                # --------------------------------------------------------------------------------
                # Latent Dynamics, Coefficient losses

                self.timer.start("Calibration");

                # Compute the latent dynamics and coefficient losses. Also fetch the current latent
                # dynamics coefficients for each training point. The latter is stored in a 2d array 
                # called "coefs" of shape (n_train, n_coefs), where n_train = number of training 
                # combinations of parameters and n_coefs denotes the number of coefficients in the 
                # latent dynamics model. 
                coefs, loss_LD, loss_coef       = LD.calibrate(Latent_States    = Latent_States, 
                                                               t_Grid           = t_Train_device,
                                                               input_coefs      = train_coefs_list,
                                                               loss_type        = self.loss_types['LD']);

                self.timer.end("Calibration");


                # ---------------------------------------------------------------------------------
                # Rollout loss. Note that we need the coefficients before we can compute this.

                # Setup
                loss_rollout_FOM    : torch.Tensor              = torch.zeros(1, dtype = torch.float32, device = device);
                loss_rollout_ROM    : torch.Tensor              = torch.zeros(1, dtype = torch.float32, device = device);
                
                if(self.loss_weights['rollout'] > 0 and p_rollout > 0):
                    self.timer.start("Rollout Loss");

                    # Simulate the frames forward in time. This should return an n_param element list
                    # whose i'th element is a 1 element list whose only element has shape (n_t_i, 
                    # n_rollout_ICs[i], n_z) whose p, q, r element of the should hold the r'th 
                    # component of the p'th frame of the j'th time derivative of the solution
                    # when we use the p'th initial condition for the i'th combination of parameter 
                    # values.
                    #
                    # Note that the latent dynamics are autonomous. Further, because we are simulating 
                    # each IC for the same amount of time, the specific values of the time are
                    # irreverent. The simulate function exploits this by solving one big IVP for each 
                    # combination of parameter values, rather than n(i) smaller ones.                         
                    ROM_Predicted_Rollout_Trajectories  : list[list[torch.Tensor]]  = self.latent_dynamics.simulate(  
                                                                                            coefs   = coefs, 
                                                                                            IC      = ROM_Rollout_ICs, 
                                                                                            t_Grid  = t_Grid_rollout);            

                    # Now cycle through the training examples.
                    for i in range(n_train):
                        # Fetch the latent displacement/velocity for the i'th combination of parameter
                        # values. 
                        ROM_Predicted_Rollout_Trajectories_i             : torch.Tensor        = ROM_Predicted_Rollout_Trajectories[i][0];            # shape = (len(t_Grid_rollout[i]), n_rollout_ICs[i], n_z)

                        # Fetch Rollout_Indices[i][j]'th frame from the rollout trajectory for the 
                        # j'th IC, which represents an approximation of the solution at 
                        # self.t_Train[i][j] + t_Grid_rollout[i][Rollout_Indices[i][j]]. We will 
                        # compare this to the interpolated FOM solution at the same time, which 
                        # should be stored in FOM_Rollout_Targets[i][0][j, ...]

                        # First, fetch the predicted solution at the target times.
                        if(self.randomized_rollout == True):
                            ROM_Rollout_Predict_i : torch.Tensor = torch.empty((n_rollout_ICs[i], model_device.n_z), dtype = torch.float32, device = device);
                            for j in range(n_rollout_ICs[i]):
                                ROM_Rollout_Predict_i[j, :] = ROM_Predicted_Rollout_Trajectories_i[Rollout_Indices[i][j], j, :];
                        else:
                            # In this case, the targets are the final frames of the rolled out 
                            # trajectories.
                            ROM_Rollout_Predict_i : torch.Tensor = ROM_Predicted_Rollout_Trajectories_i[-1, ...];


                        # Now fetch the corresponding targets.
                        ROM_Rollout_Targets_i     : list[torch.Tensor]  = ROM_Rollout_Targets[i][0];    # shape = (n_rollout_ICs[i], n_z)

                        # Decode the latent predictions to get FOM predictions.
                        FOM_Rollout_Predict_i     : torch.Tensor        = model_device.Decode(ROM_Rollout_Predict_i);
                    
                        # Get the corresponding FOM targets.
                        FOM_Rollout_Target_i      : list[torch.Tensor]  = FOM_Rollout_Targets[i][0];    # shape = (n_rollout_ICs[i], physics.Frame_Shape)
                    
                        # Compute the losses for the i'th combination of parameter values!
                        if(self.loss_types['rollout'] == "MSE"):
                            loss_rollout_ROM  += self.MSE(ROM_Rollout_Targets_i, ROM_Rollout_Predict_i);   
                            loss_rollout_FOM  += self.MSE(FOM_Rollout_Predict_i, FOM_Rollout_Target_i)/(self.std_Train[i][0]**2);
                        elif(self.loss_types['rollout'] == "MAE"):
                            loss_rollout_ROM  += self.MAE(ROM_Rollout_Targets_i, ROM_Rollout_Predict_i);   
                            loss_rollout_FOM  += self.MAE(FOM_Rollout_Predict_i, FOM_Rollout_Target_i)/self.std_Train[i][0];
                        
                    self.timer.end("Rollout Loss");


                # --------------------------------------------------------------------------------
                # IC Rollout loss. This simulates forward from the FOM initial conditions.

                loss_IC_rollout_ROM  : torch.Tensor              = torch.zeros(1, dtype = torch.float32, device = device);
                loss_IC_rollout_FOM  : torch.Tensor              = torch.zeros(1, dtype = torch.float32, device = device);

                # Cycle through the training examples for IC rollout
                if(self.loss_weights['IC_rollout'] > 0 and p_IC_rollout > 0):
                    self.timer.start("IC Rollout Loss");

                    for i in range(n_train):
                        # Fetch the FOM initial conditions for this combination of parameters
                        param_i           : numpy.ndarray             = self.param_space.train_space[i, :]; 
                        FOM_IC_i          : list[numpy.ndarray]       = self.physics.initial_condition(param_i);    # len = 1

                        # Convert to tensors and reshape for encoding
                        U_IC_i            : torch.Tensor              = torch.tensor(FOM_IC_i[0], dtype = torch.float32, device = device).reshape((1,) + FOM_IC_i[0].shape);
                        
                        # Encode the FOM initial conditions
                        Z_IC_i = model_device.Encode(U_IC_i);
                        
                        # Get the coefficients for this combination of parameters
                        coef_i            : torch.Tensor              = coefs[i, :].reshape(1, -1);
                        
                        # Simulate the latent dynamics forward in time
                        Z_IC_Rollout_i    : list[list[torch.Tensor]]  = self.latent_dynamics.simulate(  coefs   = coef_i, 
                                                                                                        IC      = [[Z_IC_i]], 
                                                                                                        t_Grid  = [t_Grid_IC_rollout[i]]);
                        
                        # Extract the predicted trajectory, remove the singleton dimension
                        Z_IC_Predict_i    : torch.Tensor              = Z_IC_Rollout_i[0][0].squeeze(1);    # shape = (n_t_IC_rollout[i], n_z)

                        # Decode the predicted trajectory to get FOM predictions
                        U_IC_Predict_i = model_device.Decode(Z_IC_Predict_i);
                        
                        # Get the corresponding FOM targets
                        U_IC_Target_i     : list[torch.Tensor]        = U_IC_Rollout_Targets[i][0];         # shape = (n_t_IC_rollout[i], physics.Frame_Shape)

                        # Encode the FOM targets for latent space comparison
                        Z_IC_Target_i = model_device.Encode(U_IC_Target_i);

                        # Compute the losses for the i'th combination of parameter values!
                        if(self.loss_types['IC_rollout'] == "MSE"):
                            loss_IC_rollout_ROM  += self.MSE(Z_IC_Target_i, Z_IC_Predict_i);
                            loss_IC_rollout_FOM  += self.MSE(U_IC_Target_i, U_IC_Predict_i)/(self.std_Train[i][0]**2);
                        elif(self.loss_types['IC_rollout'] == "MAE"):
                            loss_IC_rollout_ROM  += self.MAE(Z_IC_Target_i, Z_IC_Predict_i);
                            loss_IC_rollout_FOM  += self.MAE(U_IC_Target_i, U_IC_Predict_i)/self.std_Train[i][0];

                    self.timer.end("IC Rollout Loss");


                # --------------------------------------------------------------------------------
                # Total loss

                loss_rollout    : torch.Tensor  = loss_rollout_ROM    + loss_rollout_FOM;
                loss_IC_rollout : torch.Tensor  = loss_IC_rollout_ROM + loss_IC_rollout_FOM;


                # Compute the final loss.
                loss = (self.loss_weights['recon']      * loss_recon + 
                        self.loss_weights['LD']         * loss_LD + 
                        self.loss_weights['rollout']    * loss_rollout + 
                        self.loss_weights['IC_rollout'] * loss_IC_rollout + 
                        self.loss_weights['coef']       * loss_coef);


            # Convert coefs to numpy and find the maximum element.
            coefs           : numpy.ndarray = coefs.detach().numpy();                # Shape = (n_train, n_coefs).
            max_coef        : numpy.float32 = numpy.abs(coefs).max();


            # -------------------------------------------------------------------------------------
            # Backward Pass

            self.timer.start("Backwards Pass");

            #  Run back propagation and update the model parameters. 
            loss.backward();
            self.optimizer.step();

            # Check if we hit a new minimum loss. If so, make a checkpoint, record the loss and 
            # the iteration number. 
            if loss.item() < best_loss:
                LOGGER.info("Got a new lowest loss (%f) on epoch %d" % (loss.item(), iter + 1));
                torch.save(model_device.cpu().state_dict(), self.path_checkpoint + '/' + 'checkpoint.pt');
                
                # Update the best set of parameters. 
                self.best_coefs     = coefs.copy();       # Shape = (n_train, n_coefs).
                self.best_epoch     = iter;
                best_loss           = loss.item();

            self.timer.end("Backwards Pass");


            # -------------------------------------------------------------------------------------
            # Report Results from this iteration 

            self.timer.start("Report");

            # Report the current iteration number and losses
            if(isinstance(model_device, Autoencoder)):
                info_str : str = "Iter: %05d/%d, Total: %3.10f" % (iter + 1, self.max_iter, loss.item());
                if(self.loss_weights['recon'] > 0):         info_str += ", Recon: %3.6f"                            % loss_recon.item();
                if(self.loss_weights['rollout'] > 0):       info_str += ", Roll FOM: %3.6f, Roll ROM: %3.6f"        % (loss_rollout_FOM.item(),    loss_rollout_ROM.item());
                if(self.loss_weights['IC_rollout'] > 0):    info_str += ", IC Roll FOM: %3.6f, IC Roll ROM: %3.6f"  % (loss_IC_rollout_FOM.item(), loss_IC_rollout_ROM.item());
                if(self.loss_weights['LD'] > 0):            info_str += ", LD: %3.6f"                               % loss_LD.item();
                if(self.loss_weights['coef'] > 0):          info_str += ", Coef: %3.6f"                             % loss_coef.item();
                info_str += ", max|c|: %.3f" % max_coef;
                LOGGER.info(info_str);

            # If there are fewer than 6 training examples, report the set of parameter combinations.
            if n_train < 6:
                param_string : str  = 'Param: ' + str(numpy.round(self.param_space.train_space[0, :], 4));
                for i in range(1, n_train - 1):
                    param_string    = param_string + ', ' + str(numpy.round(self.param_space.train_space[i, :], 4));
                param_string        = param_string + ', ' + str(numpy.round(self.param_space.train_space[-1, :], 4));

                LOGGER.debug(param_string);

            # Otherwise, report the final 6 parameter combinations.
            else:
                param_string : str  = 'Param: ...';
                for i in range(5):
                    param_string    = param_string + ', ' + str(numpy.round(self.param_space.train_space[-6 + i, :], 4));
                param_string        = param_string + ', ' + str(numpy.round(self.param_space.train_space[-1, :], 4));
            
                LOGGER.debug(param_string);

            self.timer.end("Report");
            self.timer.end("train_step");
        
        # We are ready to wrap up the training procedure.
        self.timer.start("finalize");

        # Now that we have completed another round of training, update the restart iteration.
        self.restart_iter += self.n_iter;

        # Recover the model + coefficients which attained the lowest loss from this round of 
        # training.
        assert(self.best_coefs is not None);
        LOGGER.info("Model attained it's best performance on epoch %d. Replacing model with the checkpoint from that epoch" % self.best_epoch);
        state_dict  = torch.load(self.path_checkpoint + '/' + 'checkpoint.pt');
        self.model.load_state_dict(state_dict);

        # Report timing information.
        self.timer.end("finalize");
        self.timer.print();

        # All done!
        return;


    def _reset_optimizer(self) -> None:
        """
        Set the optimizer's m_t and v_t attributes (first and second moments) to zero. After each 
        training round, the momentum from the previous epoch may point us in the wrong direction. 
        Resetting the momentum eliminates this problem.
        """

        # Cycle through the optimizer's parameter groups.
        for group in self.optimizer.param_groups:

            # Cycle through the parameters in the group.
            for p in group['params']:
                state : dict = self.optimizer.state[p];

                # If the state is empty, skip this parameter.
                if not state:
                    continue;
                
                # zero the biased first moment estimate
                state['exp_avg'].zero_();

                # zero the biased second moment estimate
                state['exp_avg_sq'].zero_();
                
                # if you're using amsgrad:
                if 'max_exp_avg_sq' in state:
                    state['max_exp_avg_sq'].zero_();


    
    def _rollout_setup( self, 
                        t           : list[torch.Tensor], 
                        U           : list[list[torch.Tensor]], 
                        p_rollout   : float) -> tuple[list[torch.Tensor], list[int], list[list[torch.Tensor]]]:
        """
        An internal function that sets up the rollout loss. Specifically, it finds the t_grid for 
        simulating each latent frame we plan to rollout, the number of frames we can rollout for 
        each parameter value, the final time of each frame we rollout, and a target FOM frame for 
        each frame we rollout. The user should not call this function directly; only the train 
        method should call this.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        t: : list[torch.Tensor], len = n_param
            i'th element is a 1d torch.Tensor of shape (n_t_i) whose j'th element specifies the 
            time of the j'th frame in the FOM solution for the i'th combination of parameter 
            values.

        U : list[list[torch.Tensor]], len = n_param
            i'th element is a n_IC element list whose j'th element is a torch.Tensor of shape 
            (n_t_i, ...) whose k'th element specifies the value of the j'th time derivative of the 
            FOM frame when using the i'th combination of parameter values.

        p_rollout : float
            A number between 0 and 1 specifying the ratio of the rollout time for a particular 
            combination of parameter values to the length of the time interval for that combination 
            of parameter values.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        t_Grid_rollout, n_rollout_ICs, U_Target_Rollout_Trajectory

        t_Grid_rollout : list[torch.Tensor], len = n_param
            i'th element is a 1d array whose j'th entry holds the j'th time at which we want to 
            rollout the first frame for the i'th combination of parameter values. Why do we do 
            this? When we rollout the latent states, we take advantage of the fact that the 
            governing dynamics is autonomous. Specifically, the actual at which times we solve the 
            ODE do not matter. All that matters is amount of time we solve for. This allows us to 
            use the same time grid for all latent states that we rollout, which dramatically 
            improves runtime. 

        n_rollout_ICs : list[int], len = n_param
            i'th element specifies how many frames we can rollout from the FOM solution for the 
            i'th combination of parameter values. Specifically, the first n_rollout_ICs[i] 
            frames from the i'th FOM solution are such that the time for each frame after rollout 
            will be less than the final time for that FOM solution.

        U_Target_Rollout_Trajectory : list[list[torch.Tensor]], len = n_param
            i'th element is an n_IC element list whose j'th element is a torch.Tensor of shape 
            (n_rollout_ICs[i], n_rollout_steps[i], physics.Frame_Shape), where n_rollout_steps[i] is 
            len(t_Grid_rollout[i]). The U_Target_Rollout_Trajectory[i][j][k, l] holds the 
            interpolated j'th derivative of the FOM solution at the l'th time step of the rollout 
            for the k'th IC for the i'th combination of parameter values. 
        """

        # Checks
        assert isinstance(p_rollout, float),    "type(p_rollout) = %s" % str(type(p_rollout));
        assert isinstance(U, list),             "type(U) = %s" % str(type(U));
        assert isinstance(t, list),             "type(t) = %s" % str(type(t));
        assert len(t) == len(U),                "len(t) = %d, len(U) = %d" % (len(t), len(U));

        assert isinstance(U[0], list),          "type(U[0]) = %s" % str(type(U[0]));
        n_param     : int   = len(U);

        for i in range(n_param):
            assert isinstance(U[i], list),          "type(U[%d]) = %s" % (i, str(type(U[i])));
            assert isinstance(t[i], torch.Tensor),  "type(t[%d]) = %s" % (i, str(type(t[i])));
            assert len(U[i])        == self.n_IC,   "len(U[%d]) = %d, self.n_IC = %d" % (i, len(U[i]), self.n_IC);
            assert len(t[i].shape)  == 1,           "len(t[%d].shape) = %d" % (i, len(t[i].shape));

            n_t_i : int = t[i].shape[0];
            for j in range(self.n_IC):
                assert isinstance(U[i][j], torch.Tensor), "type(U[%d][%d]) = %s" % (i, j, str(type(U[i][j])));
                assert U[i][j].shape[0]     == n_t_i,     "U[%d][%d].shape[0] = %d, n_t_i = %d" % (i, j, U[i][j].shape[0], n_t_i);


        # Other setup.        
        t_Grid_rollout                  : list[torch.Tensor]        = [];   # n_train element list whose i'th element is 1d array of times for rollout solve.
        n_rollout_ICs                   : list[int]                 = [];   # n_train element list whose i'th element specifies how many frames we should simulate forward.
        U_Target_Rollout_Trajectory   : list[list[torch.Tensor]]  = [];   # n_train element list whose i'th element is n_IC element list whose j'th element is a torch.Tensor of shape (n_rollout_ICs[i], n_rollout_steps[i], physics.Frame_Shape) holding target trajectories when we rollout the IC's for the j'th time derivative/i'th combination of parameters. 


        # -----------------------------------------------------------------------------------------
        # Find t_Grid_rollout, and n_rollout_ICs.

        for i in range(n_param):
            # Determine the amount of time that passes in the FOM simulation corresponding to the 
            # i'th combination of parameter values. 
            t_i                 : torch.Tensor  = t[i];
            n_t_i               : int           = t_i.shape[0];
            t_0_i               : float         = t_i[0].item();
            t_final_i           : float         = t_i[-1].item();

            # The final rollout time for this combination of parameter values. Remember that 
            # t_rollout is the proportion of t_final_i - t_0_i over which we simulate.
            t_rollout_i         : float         = p_rollout*(t_final_i - t_0_i);
            t_rollout_final_i   : float         = t_rollout_i + t_0_i;
            LOGGER.info("We will rollout the first frame for parameter combination #%d to t = %f" % (i, t_rollout_final_i));

            # Now figure out how many time steps occur before t_rollout_final_i.
            num_before_rollout_final_i  : int           = 0;
            for j in range(n_t_i):
                if(t_i[j] > t_rollout_final_i):
                    break; 
                
                num_before_rollout_final_i += 1;
            LOGGER.info("We will rollout each frame for parameter combination #%d over %d time steps" % (i, num_before_rollout_final_i));


            # Now define the rollout time grid for the i'th combination of parameter values.
            t_Grid_rollout.append(torch.linspace(start = t_0_i, end = t_rollout_final_i, steps = num_before_rollout_final_i));

            # Now figure out how many times occur less than t_rollout_final_i from t_final_i.
            n_rollout_ICs_i : int = 0;
            for j in range(n_t_i):
                if(t_i[j] + t_rollout_i > t_final_i):
                    break;

                n_rollout_ICs_i += 1;
            n_rollout_ICs.append(n_rollout_ICs_i);
            LOGGER.info("We will rollout %d FOM frames for parameter combination #%d." % (n_rollout_ICs_i, i));


        # -----------------------------------------------------------------------------------------
        # Find U_Target_Rollout_Trajectory.

        for i in range(n_param):
            LOGGER.debug("Making interpolators for parameter combination #%d" % i);

            # Interpolate each U_Train[i][j], then evaluate it at the target times.
            U_Train_i               : list[torch.Tensor]            = U[i];                         # len = n_IC, i'th element is a torch.Tensor of shape (n_t(i), ...)
            t_Train_i               : numpy.ndarray                 = t[i].detach().numpy();        # shape = (n_t(i))

            # Fetch the number of frames we will rollout and the number of time 
            # steps we will rollout.
            n_rollout_ICs_i         : int = n_rollout_ICs[i];
            n_rollout_steps_i       : int = len(t_Grid_rollout[i]);

            # Fetch the targets for the i'th combination of parameter values.
            U_Target_Rollout_Trajectory_i       : list[torch.Tensor]            = [];
            for j in range(self.n_IC):
                # Interpolate the time series for the j'th derivative of the FOM solution when we 
                # use the i'th combination of parameter values.
                U_Train_ij          : numpy.ndarray = U_Train_i[j].detach().numpy();        # shape = (n_t(i), Physics.Frame_Shape)
                U_Train_ij_interp                   = interpolate.make_interp_spline(x = t_Train_i, y = U_Train_ij, k = self.rollout_spline_order);

                U_Target_Rollout_Trajectory_ij : numpy.ndarray = numpy.empty((n_rollout_ICs_i, n_rollout_steps_i) + tuple(self.physics.Frame_Shape), dtype = numpy.float32);
                for k in range(n_rollout_ICs_i):
                    # Evaluate the i,j interpolator at the target times for the k'th rollout IC.
                    # The target times for the k'th IC rollout are the rollout timnes for the 1st 
                    # frame (t_Grid_rollout) plus the time of the k'th IC (t_Train_i[k]).
                    U_Target_Rollout_Trajectory_ij[k, ...] = U_Train_ij_interp(t_Grid_rollout[i] + t_Train_i[k]*numpy.ones_like(t_Grid_rollout[i], dtype = numpy.float32));

                # Evaluate the interpolation at the final rollout times for the i'th combination of
                # parameter values.
                U_Target_Rollout_Trajectory_i.append(torch.tensor(U_Target_Rollout_Trajectory_ij, dtype = torch.float32, device = self.device));
            U_Target_Rollout_Trajectory.append(U_Target_Rollout_Trajectory_i);
    

        # All done!
        return t_Grid_rollout, n_rollout_ICs, U_Target_Rollout_Trajectory;



    def _IC_rollout_setup( self, 
                           t            : list[torch.Tensor], 
                           p_IC_rollout : float) -> tuple[list[torch.Tensor], list[int], list[list[torch.Tensor]]]:
        """
        An internal function that sets up the IC rollout loss. This is similar to _rollout_setup but
        for simulating forward from the FOM initial conditions. The user should not call this 
        function directly; only the train method should call this.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        t : list[torch.Tensor], len = n_param
            i'th element is a 1d torch.Tensor of shape (n_t_i) whose j'th element specifies the 
            time of the j'th frame in the FOM solution for the i'th combination of parameter 
            values.

        p_IC_rollout : float
            A number between 0 and 1 specifying the ratio of the IC rollout time for a particular 
            combination of parameter values to the length of the time interval for that combination 
            of parameter values.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        t_Grid_IC_rollout, n_IC_rollout_frames, U_IC_Rollout_Targets

        t_Grid_IC_rollout : list[torch.Tensor], len = n_param
            i'th element is a 1d array whose j'th entry holds the j'th time at which we want to 
            rollout the initial condition for the i'th combination of parameter values.

        n_IC_rollout_frames : list[int], len = n_param
            i'th element specifies how many time steps we simulate forward from the initial condition
            for the i'th combination of parameter values.

        U_IC_Rollout_Targets : list[list[torch.Tensor]], len = n_param
            i'th element is an n_IC element list whose j'th element is a torch.Tensor of shape 
            (n_IC_rollout_frames[i], physics.Frame_Shape) consisting of the first 
            n_IC_rollout_frames[i] frames of the j'th time derivative of the FOM solution for the 
            i'th combination of parameter values.
        """

        # Checks
        assert isinstance(p_IC_rollout, float), "type(p_IC_rollout) = %s" % str(type(p_IC_rollout));
        assert isinstance(t, list),             "type(t) = %s" % str(type(t));
        assert p_IC_rollout >= 0.0 and p_IC_rollout <= 1.0, "p_IC_rollout = %f" % p_IC_rollout;

        n_param     : int   = len(t);

        # Other setup.        
        t_Grid_IC_rollout          : list[torch.Tensor]         = [];   # n_train element list whose i'th element is 1d array of times for IC rollout solve.
        n_IC_rollout_frames        : list[int]                  = [];   # n_train element list whose i'th element specifies how many time steps we should simulate forward.
        U_IC_Rollout_Targets       : list[list[torch.Tensor]]   = [];   # n_train element list whose i'th element is n_IC element list whose j'th element is a tensor of shape (n_IC_rollout_frames[i], ...) specifying FOM IC rollout targets


        # -----------------------------------------------------------------------------------------
        # Find t_Grid_IC_rollout and n_IC_rollout_frames.

        for i in range(n_param):
            # Determine the amount of time that passes in the FOM simulation corresponding to the 
            # i'th combination of parameter values. 
            t_i                 : torch.Tensor  = t[i];
            n_t_i               : int           = t_i.shape[0];
            t_0_i               : float         = t_i[0].item();
            t_final_i           : float         = t_i[-1].item();

            # The final IC rollout time for this combination of parameter values. Remember that 
            # t_IC_rollout is the proportion of t_final_i - t_0_i over which we simulate.
            t_IC_rollout_i      : float         = p_IC_rollout*(t_final_i - t_0_i);
            t_IC_rollout_final_i: float         = t_IC_rollout_i + t_0_i;
            LOGGER.info("We will rollout the initial condition for parameter combination #%d to t = %f" % (i, t_IC_rollout_final_i));

            # Now figure out how many time steps occur before t_IC_rollout_final_i.
            num_before_IC_rollout_final_i  : int           = 0;
            for j in range(n_t_i):
                if(t_i[j] > t_IC_rollout_final_i):
                    break; 
                
                num_before_IC_rollout_final_i += 1;
            LOGGER.info("We will rollout the initial condition for parameter combination #%d over %d time steps" % (i, num_before_IC_rollout_final_i));

            # Now define the IC rollout time grid for the i'th combination of parameter values.
            t_Grid_IC_rollout.append(torch.linspace(start = t_0_i, end = t_IC_rollout_final_i, steps = num_before_IC_rollout_final_i));

            # The number of frames we simulate forward from the initial condition
            n_IC_rollout_frames.append(num_before_IC_rollout_final_i);
            LOGGER.info("We will simulate %d time steps from the initial condition for parameter combination #%d." % (num_before_IC_rollout_final_i, i));

            # Fetch the first n_IC_rollout_frames[i] FOM frames.
            U_IC_Rollout_Targets_i : list[torch.Tensor] = [];
            for j in range(self.n_IC):
                U_IC_Rollout_Targets_i.append(self.U_Train[i][j][:num_before_IC_rollout_final_i]);
            U_IC_Rollout_Targets.append(U_IC_Rollout_Targets_i);

        # All done!
        return t_Grid_IC_rollout, n_IC_rollout_frames, U_IC_Rollout_Targets;



    def get_new_sample_point(self) -> numpy.ndarray:
        """
        This function finds the element of the testing set (excluding the training set) whose 
        corresponding latent dynamics gives the highest variance FOM time series. 

        How does this work? The latent space coefficients change with parameter values. For each 
        coefficient, we fit a gaussian process whose input is the parameter values. Thus, for each 
        potential parameter value and coefficient, we can find a distribution for that coefficient 
        when we use that parameter value.

        With this in mind, for each combination of parameters in self.param_space's test space, 
        we draw a set of samples of the coefficients at that combination of parameter values. For
        each combination, we solve the latent dynamics forward in time (using the sampled set of
        coefficient values to define the latent dynamics). This gives us a time series of latent 
        states. We do this for each sample, for each testing parameter. 

        For each time step and parameter combination, we get a set of latent frames. We map that 
        set to a set of FOM frames and then find the STD of each component of those FOM frames 
        across the samples. This give us a number. We find the corresponding number for each time 
        step and combination of parameter values and then return the parameter combination that 
        gives the biggest number (for some time step). This becomes the new sample point.

        Thus, the sample point is ALWAYS an element of the testing set. 



        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        None!

        

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        new_sample : numpy.ndarray, shape = (1, n_p)
            (0, j) element holds the value of the j'th parameter in the new sample. Here, n_p is 
            the number of parameters.
        """

        self.timer.start("new_sample");
        assert len(self.U_Test)             >  0,                           "len(self.U_Test) = %d" % len(self.U_Test);
        assert len(self.U_Test)             == self.param_space.n_test(),   "len(self.U_Test) = %d, self.param_space.n_test() = %d" % (len(self.U_Test), self.param_space.n_test());
        assert self.best_coefs.shape[0]     == self.param_space.n_train(),  "self.best_coefs.shape[0] = %d, self.param_space.n_train() = %d" % (self.best_coefs.shape[0], self.param_space.n_train());

        coefs : numpy.ndarray = self.best_coefs;                        # Shape = (n_train, n_coefs).
        LOGGER.info('\n~~~~~~~ Finding New Point ~~~~~~~');

        # Move the model to the cpu (this is where all the GP stuff happens) and load the model 
        # from the last checkpoint. This should be the one that obtained the best loss so far. 
        # Remember that coefs should specify the coefficients from that iteration. 
        model       : torch.nn.Module   = self.model.cpu();
        n_test      : int               = self.param_space.n_test();
        n_train     : int               = self.param_space.n_train();
        model.load_state_dict(torch.load(self.path_checkpoint + '/' + 'checkpoint.pt'));

        # First, find the candidate parameters. This is the elements of the testing set that 
        # are not already in the training set.
        candidate_parameters    : list[numpy.ndarray]   = [];
        t_Candidates            : list[torch.Tensor]    = [];
        for i in range(n_test):
            ith_Test_param = self.param_space.test_space[i, :];
            
            # Check if the i'th testing parameter is in the training set
            in_train : bool = False;
            for j in range(n_train):
                if(numpy.any(numpy.all(self.param_space.train_space[j, :] == ith_Test_param))):
                    in_train = True;
                    break;
            
            # If not, add it to the set of candidates
            if(in_train == False):
                candidate_parameters.append(ith_Test_param);
                t_Candidates.append(self.t_Test[i]);
        
        # Concatenate the candidates to form an array of shape (n_candidates, n_coefs).
        n_candidates : int = len(candidate_parameters);
        LOGGER.info("There are %d candidate testing parameters (%d in the testing space, %d in the training set)" % (n_candidates, n_test, n_train));
        assert n_candidates >= 1, "n_candidates = %d" % n_candidates;
        candidate_parameters    = numpy.array(candidate_parameters);


        # Map the initial conditions for the FOM to initial conditions in the latent space.
        # Yields an n_candidates element list whose i'th element is an n_IC element list whose j'th
        # element is an numpy.ndarray of shape (1, n_z) whose k'th element holds the k'th component
        # of the encoding of the initial condition for the j'th derivative of the latent dynamics 
        # corresponding to the i'th candidate combination of parameter values.
        Z0 : list[list[numpy.ndarray]]  = model.latent_initial_conditions(  param_grid  = candidate_parameters, 
                                                                            physics     = self.physics);

        # Train the GPs on the training data, get one GP per latent space coefficient.
        gp_list : list[GaussianProcessRegressor] = fit_gps(self.param_space.train_space, coefs);

        # For each combination of parameter values in the candidate set, for each coefficient, 
        # draw a set of samples from the posterior distribution for that coefficient evaluated at
        # the candidate parameters. We store the samples for a particular combination of parameter 
        # values in a 2d numpy.ndarray of shape (n_sample, n_coef), whose i, j element holds the 
        # i'th sample of the j'th coefficient. We store the arrays for different parameter values 
        # in a list of length n_test. 
        coef_samples : list[numpy.ndarray] = [sample_coefs(gp_list, candidate_parameters[i, :], self.n_samples) for i in range(n_candidates)];

        # Now, solve the latent dynamics forward in time for each set of coefficients in 
        # coef_samples. There are n_test combinations of parameter values, and we have n_samples 
        # sets of coefficients for each combination of parameter values. For the i'th one of those,
        # we want to solve the latent dynamics for n_t(i) times steps. Each solution frame consists
        # of n_IC elements of \marthbb{R}^{n_z}.
        # 
        # Thus, we store the latent states in an n_test element list whose i'th element is an n_IC
        # element list whose j'th element is an array of shape (n_samples, n_t(i), n_z) whose
        # p, q, r element holds the r'th component of j'th derivative of the latent state at the 
        # q'th time step when we use the p'th set of coefficient values sampled from the posterior
        # distribution for the i'th combination of testing parameter values.
        LatentStates    : list[list[numpy.ndarray]]     = [];
        n_z             : int                           = self.latent_dynamics.n_z;
        for i in range(n_candidates):
            LatentStates_i  : list[numpy.ndarray]    = [];
            for j in range(self.n_IC):
                LatentStates_i.append(numpy.ndarray([self.n_samples, len(self.t_Test[j]), n_z]));
            LatentStates.append(LatentStates_i);
        
        for i in range(n_candidates):
            # Fetch the t_Grid for the i'th combination of parameter values.
            t_Grid  : numpy.ndarray = t_Candidates[i].reshape(1, -1).detach().numpy();
            n_t_j   : int           = len(t_Grid);

            # Simulate one sample at a time; store the resulting frames.           
            for j in range(self.n_samples):
                LatentState_ij : list[list[numpy.ndarray]] = self.latent_dynamics.simulate( coefs   = coef_samples[i][j:(j + 1), :], 
                                                                                            IC      = [Z0[i]], 
                                                                                            t_Grid  = [t_Grid]);
                for k in range(self.n_IC):
                    LatentStates[i][k][j, :, :] = LatentState_ij[0][k][:, 0, :];

        # Find the index of the parameter with the largest std.
        m_index : int = get_FOM_max_std(model, LatentStates);

        # We have found the testing parameter we want to add to the training set. Fetch it, then
        # stop the timer and return the parameter. 
        new_sample : numpy.ndarray = candidate_parameters[m_index, :].reshape(1, -1);
        LOGGER.info('New param: ' + str(numpy.round(new_sample, 4)) + '\n');
        self.timer.end("new_sample");

        # All done!
        return new_sample;



    def export(self) -> dict:
        """
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        dict_ : dict
            A dictionary housing most of the internal variables in self. You can pass this 
            dictionary to self (after initializing it using ParameterSpace, model, and 
            LatentDynamics objects) to make a GLaSDI object whose internal state matches that of 
            self.
        """

        dict_ = {'U_Train'                  : self.U_Train, 
                 'U_Test'                   : self.U_Test,
                 't_Train'                  : self.t_Train,
                 't_Test'                   : self.t_Test,
                 'best_coefs'               : self.best_coefs,                      # Shape = (n_train, n_coefs).
                 'max_iter'                 : self.max_iter, 
                 'restart_iter'             : self.restart_iter, 
                 'timer'                    : self.timer.export(), 
                 'test_coefs'               : self.test_coefs,
                 'optimizer'                : self.optimizer.state_dict()};
        return dict_;



    def load(self, dict_ : dict) -> None:
        """
        Modifies self's internal state to match the one whose export method generated the dict_ 
        dictionary.


        -------------------------------------------------------------------------------------------
        Arguments 
        -------------------------------------------------------------------------------------------

        dict_ : dict 
            This should be a dictionary returned by calling the export method on another 
            GLaSDI object. We use this to make self hav the same internal state as the object that 
            generated dict_. 
            

        -------------------------------------------------------------------------------------------
        Returns  
        -------------------------------------------------------------------------------------------
        
        Nothing!
        """

        # Extract instance variables from dict_.
        self.U_Train        : list[list[torch.Tensor]]  = dict_['U_Train'];             # len = n_train, i'th element is an n_IC element list.  
        self.U_Test         : list[list[torch.Tensor]]  = dict_['U_Test'];              # len = n_test, i'th element is an n_IC element list.

        self.t_Train        : list[torch.Tensor]        = dict_['t_Train'];             # len = n_train.
        self.t_Test         : list[torch.Tensor]        = dict_['t_Test'];              # len = n_test.

        self.best_coefs     : numpy.ndarray             = dict_['best_coefs'];          # Shape = (n_train, n_coefs).
        self.best_epoch     : int                       = dict_['restart_iter'];        # The current model has the best loss so far.
        self.restart_iter   : int                       = dict_['restart_iter'];

        # Now compute the std of the FOM solution for each combination of training parameters.
        self.std_Train    : list[list[float]] = [];
        for i in range(len(self.U_Train)):
            self.std_Train.append([]);
            for j in range(len(self.U_Train[i])):
                self.std_Train[i].append(torch.std(self.U_Train[i][j]));

        # Next, compute n_IC.           
        self.n_IC = len(self.U_Test[0]);

        # Set the test coefs.
        with torch.no_grad():
            for i in range(len(self.test_coefs)):
                self.test_coefs[i] = dict_['test_coefs'][i];

        # Load the timer / optimizer. 
        self.timer.load(dict_['timer']);
        self.optimizer.load_state_dict(dict_['optimizer']);
        if (self.device != 'cpu'):
            optimizer_to(self.optimizer, self.device);

        # All done!
        return;
    