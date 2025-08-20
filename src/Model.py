# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add the Physics directory to the search path.
import  sys;
import  os;
Physics_Path    : str  = os.path.abspath(os.path.join(os.path.dirname(__file__), "Physics"));
sys.path.append(Physics_Path);

import  logging;
from    typing      import  Callable;

import  torch;
import  numpy;

from    Physics     import  Physics;


# Set up logging.
LOGGER  : logging.Logger    = logging.getLogger(__name__);


# activation dict
act_dict = {'elu'           : torch.nn.functional.elu,
            'hardshrink'    : torch.nn.functional.hardshrink,
            'hardsigmoid'   : torch.nn.functional.hardsigmoid,
            'hardtanh'      : torch.nn.functional.hardtanh,
            'hardswish'     : torch.nn.functional.hardswish,
            'leakyReLU'     : torch.nn.functional.leaky_relu,
            'logsigmoid'    : torch.nn.functional.logsigmoid,
            'prelu'         : torch.nn.functional.prelu,
            'relu'          : torch.nn.functional.relu,
            'relu6'         : torch.nn.functional.relu6,
            'rrelu'         : torch.nn.functional.rrelu,
            'selu'          : torch.nn.functional.selu,
            'celu'          : torch.nn.functional.celu,
            'sin'           : torch.sin,
            'cos'           : torch.cos,
            'gelu'          : torch.nn.functional.gelu,
            'sigmoid'       : torch.nn.functional.sigmoid,
            'silu'          : torch.nn.functional.silu,
            'mish'          : torch.nn.functional.mish,
            'softplus'      : torch.nn.functional.softplus,
            'softshrink'    : torch.nn.functional.softshrink,
            'tanh'          : torch.nn.functional.tanh,
            'tanhshrink'    : torch.nn.functional.tanhshrink};




# -------------------------------------------------------------------------------------------------
# MLP class
# -------------------------------------------------------------------------------------------------

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(   self, 
                    widths          : list[int],
                    activations     : list[str],
                    reshape_index   : int           = 1, 
                    reshape_shape   : list[int]     = []) -> None:
        r"""
        This class defines a standard multi-layer network network.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        widths : list[int]
            A list of integers specifying the widths of the layers (including the 
            dimensionality of the domain of each layer, as well as the co-domain of the final 
            layer). Suppose this list has N elements. Then the network will have N - 1 layers. 
            The i'th layer maps from \mathbb{R}^{widths[i]} to \mathbb{R}^{widths[i + 1]}. Thus, 
            the i'th element of this list represents the domain of the i'th layer AND the 
            co-domain of the i-1'th layer.

        activations : list[str], len = len(widths) - 2
            A list of strings whose i'th element specifies the activation function we want to use 
            after the i'th layer's linear transformation. The final layer has no activation 
            function. 

        reshape_index : int, optional
            This argument specifies if we should reshape the network's input or output 
            (or neither). If the user specifies reshape_index, then it must be either 0 or -1. 
            Further, in this case, they must also specify reshape_shape (you need to specify both 
            together). If it is 0, then reshape_shape specifies how we reshape the input before 
            passing it through the network (the input to the first layer). If reshape_index is -1, 
            then reshape_shape specifies how we reshape the network output before returning it 
            (the output to the last layer). 

        reshape_shape : list[int], optional
            This is a list of k integers specifying the final k dimensions of the shape of the 
            input to the first layer (if reshape_index == 0) or the output of the last layer 
            (if reshape_index == -1). You must specify this argument if and only if you specify 
            reshape_index. 



        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        Nothing!
        """

        # Checks
        assert(isinstance(reshape_shape, list));
        for i in range(len(reshape_shape)):
            assert(isinstance(reshape_shape[i], int));
            assert(reshape_shape[i] > 0);
        assert(isinstance(widths, list));
        assert(isinstance(activations, list));
        assert(len(activations) == len(widths) - 2);
        for i in range(len(widths) - 2):
            assert(isinstance(widths[i], int));
            assert(widths[i] > 0);
            assert(activations[i].lower() in act_dict.keys());
        assert(isinstance(reshape_shape, list));
        assert(isinstance(reshape_index, int));
        assert((len(reshape_shape) == 0) or (reshape_index in [0, -1]));
        assert((len(reshape_shape) == 0) or (numpy.prod(reshape_shape) == widths[reshape_index]));

        super().__init__();

        # Note that width specifies the dimensionality of the domains and co-domains of each layer.
        # Specifically, the i'th element specifies the input dimension of the i'th layer, while 
        # the final element specifies the dimensionality of the co-domain of the final layer. Thus, 
        # the number of layers is one less than the length of widths.
        self.n_layers       : int                   = len(widths) - 1;
        self.widths         : list[int]             = widths;
        self.activations    : list[str]             = activations;

        # Set up the affine parts of the layers.
        self.layers = [];
        for k in range(self.n_layers):
            self.layers += [torch.nn.Linear(widths[k], widths[k + 1])];
        self.layers = torch.nn.ModuleList(self.layers);

        # Now, initialize the weight matrices and bias vectors in the affine portion of each layer.
        self.init_weight();

        # Reshape input to the 1st layer or output of the last layer.
        self.reshape_index : int        = reshape_index;
        self.reshape_shape : list[int]  = reshape_shape;

        # Set up the activation functions
        self.activation_fns : list[Callable] = [];
        for i in range(self.n_layers - 1):
            self.activation_fns.append(act_dict[self.activations[i].lower()]);

        LOGGER.info("Initializing a MultiLayerPerceptron with widths %s, activations %s, reshape_shape = %s (index %d)" \
                    % (str(self.widths), str(self.activations), str(self.reshape_shape), self.reshape_index));

        # All done!
        return;
    


    def forward(self, U : torch.Tensor) -> torch.Tensor:
        """
        This function defines the forward pass through self.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        X : torch.Tensor
            A tensor holding a batch of inputs. We pass this tensor through the network's layers 
            and then return the result. If self.reshape_index == 0 and self.reshape_shape has k
            elements, then the final k elements of X's shape must match self.reshape_shape. 


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        U_Pred : torch.Tensor, shape = X.Shape
            The image of X under the network's layers. If self.reshape_index == -1 and 
            self.reshape_shape has k elements, then we reshape the output so that the final k 
            elements of its shape match those of self.reshape_shape.
        """

        # If the reshape_index is 0, we need to reshape X before passing it through the first 
        # layer.
        if ((len(self.reshape_shape) > 0) and (self.reshape_index == 0)):
            # Make sure the input has a proper shape. There is a lot going on in this line; let's
            # break it down. If reshape_index == 0, then we need to reshape the input, X, before
            # passing it through the layers. Let's assume that reshape_shape has k elements. Then,
            # we need to squeeze the final k dimensions of the input, X, so that the resulting 
            # tensor has a final dimension size that matches the input dimension size for the first
            # layer. The check below makes sure that the final k dimensions of the input, X, match
            # the stored reshape_shape.
            assert(list(U.shape[-len(self.reshape_shape):]) == self.reshape_shape);
            
            # Now that we know the final k dimensions of X have the correct shape, let's squeeze 
            # them into 1 dimension (so that we can pass the squeezed tensor through the first 
            # layer). To do this, we reshape X by keeping all but the last k dimensions of X, and 
            # replacing the last k with a single dimension whose size matches the dimensionality of
            # the domain of the first layer. 
            U = U.reshape(list(U.shape[:-len(self.reshape_shape)]) + [self.widths[self.reshape_index]]);

        # Pass X through the network layers; note that the final layer has no activation function, 
        # so we don't apply an activation function to it.
        for i in range(self.n_layers - 1):
            U = self.activation_fns[i](self.layers[i](U));   # apply linear layer
        U = self.layers[-1](U);                              # apply final layer (no activation)

        # If the reshape_index is -1, then we need to reshape the output before returning. 
        if ((len(self.reshape_shape) > 0) and (self.reshape_index == -1)):
            # In this case, we need to split the last dimension of X, the output of the final
            # layer, to match the reshape_shape. This is precisely what the line below does. Note
            # that we use torch.Tensor.view instead of torch.Tensor.reshape in order to avoid data 
            # copying. 
            U = U.view(list(U.shape[:-1]) + self.reshape_shape);

        # All done!
        return U;
    


    def init_weight(self) -> None:
        """
        This function initializes the weight matrices and bias vectors in self's layers. It takes 
        no arguments and returns nothing!
        """

        # TODO(kevin): support other initializations?
        for layer in self.layers:
            torch.nn.init.xavier_uniform_(layer.weight);
            torch.nn.init.zeros_(layer.bias);
        
        # All done!
        return



# -------------------------------------------------------------------------------------------------
# Autoencoder class
# -------------------------------------------------------------------------------------------------

class Autoencoder(torch.nn.Module):
    def __init__(   self,                     
                    reshape_shape   : list[int],
                    widths          : list[int], 
                    activations     : list[str]) -> None:
        r"""
        Initializes an Autoencoder object. An Autoencoder consists of two networks, an encoder, 
        E : \mathbb{R}^F -> \mathbb{R}^L, and a decoder, D : \mathbb{R}^L -> \marthbb{R}^F. We 
        assume that the dataset consists of samples of a parameterized L-manifold in 
        \mathbb{R}^F. The idea then is that E and D act like the inverse coordinate patch and 
        coordinate patch, respectively. In our case, E and D are trainable neural networks. We 
        try to train E and map data in \mathbb{R}^F to elements of a low dimensional latent 
        space (\mathbb{R}^L) which D can send back to the original data. (thus, E, and D should
        act like inverses of one another).

        The Autoencoder class implements this model as a trainable torch.nn.Module object. 


        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        reshape_shape : list[int]
            This is a list of k integers specifying the final k dimensions of the shape of the 
            input to the first layer (if reshape_index == 0) or the output of the last layer (if 
            reshape_index == -1). 
        
        widths : list[int]
            A list of integers specifying the widths of the layers in the encoder. We use the 
            revere of this list to specify the widths of the layers in the decoder. See the 
            docstring for the MultiLayerPerceptron class for details on how Widths defines a 
            network.

        activations : list[str], len = len(widths) - 2
            i'th element specifies which activation function we want to use after the i'th layer 
            in the encoder. The final layer has no activation function. We use the reversed list
            for the decoder. 



        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Checks
        assert(isinstance(reshape_shape, list));
        for i in range(len(reshape_shape)):
            assert(isinstance(reshape_shape[i], int));
            assert(reshape_shape[i] > 0);
        assert(isinstance(widths, list));
        for i in range(len(widths)):
            assert(isinstance(widths[i], int));
            assert(widths[i] > 0);
        assert(isinstance(activations, list));
        assert(len(activations) == len(widths) - 2);
        for i in range(len(activations)):
            assert(activations[i].lower() in act_dict.keys());

        # Run the superclass initializer.
        super().__init__();
        
        # Store information (for return purposes).
        self.n_IC           : int       = 1;
        self.widths         : list[int] = widths;
        self.n_z            : int       = widths[-1];
        self.activations    : list[str] = activations;
        self.reshape_shape  : list[int] = reshape_shape;
        LOGGER.info("Initializing an Autoencoder with latent space dimension %d" % self.n_z);

        # Build the encoder, decoder.
        LOGGER.info("Initializing the encoder...");
        self.encoder = MultiLayerPerceptron(
                            widths              = widths, 
                            activations         = activations,
                            reshape_index       = 0,                    # We need to flatten the spatial dimensions of each FOM frame.
                            reshape_shape       = reshape_shape);

        LOGGER.info("Initializing the decoder...");
        self.decoder = MultiLayerPerceptron(
                            widths              = widths[::-1],         # Reverses the order for the decoder.
                            activations         = activations[::-1],    # Reverses the order for the decoder.
                            reshape_index       = -1,                   # We need to reshape the network output to a FOM frame.
                            reshape_shape       = reshape_shape);       # We need to reshape the network output to a FOM frame.


        # All done!
        return;




    def Encode(self, U : torch.Tensor) -> torch.Tensor:
        """
        This function encodes a set of displacement and velocity frames.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        X : torch.Tensor, shape = (n_Frames,) + self.reshape_shape
            X[i, ...] holds the i'th frame we want to encode. 


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Z : torch.Tensor, shape = (n_Frasmes, self.n_z)
            i,j element holds the j'th component of the encoding of the i'th FOM frame.
        """

        # Check that the inputs have the correct shape.
        assert(len(U.shape)         ==  len(self.reshape_shape) + 1,    "U.shape = %s, self.reshape_shape = %s"     % (str(U.shape), str(self.reshape_shape)));
        assert(list(U.shape[1:])    ==  self.reshape_shape,             "U.shape[1:] = %s, self.reshape_shape = %s" % (str(U.shape[1:]), str(self.reshape_shape)));
    
        # Encode the frames!
        return self.encoder(U);



    def Decode(self, Z : torch.Tensor)-> torch.Tensor:
        """
        This function decodes a set of latent frames.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Z : torch.Tensor, shape = (n_Frames, self.n_z)
           i,j element holds the j'th component of the encoding of the i'th frame.
     

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        R : torch.Tensor, shpe = (n_Frames,) + self.reshape_shape
            R[i ...] represents the reconstruction of the i'th FOM frame.
        """

        # Check that the input has the correct shape. 
        assert(len(Z.shape)   == 2);
    
        # Decode the frames!
        return self.decoder(Z);




    def forward(self, U : torch.Tensor) -> torch.Tensor:
        """
        This function passes X through the encoder, producing a latent state, Z. It then passes 
        Z through the decoder; hopefully producing a vector that approximates X.
        

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        U : torch.Tensor, shape = (n_Frames,) + self.reshape_shape
            A tensor holding a batch of inputs. We pass this tensor through the encoder + decoder 
            and then return the result.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Y : torch.Tensor, shape = X.shape
            The image of X under the encoder and decoder. 
        """

        # Encoder the input
        Z : torch.Tensor    = self.Encode(U);

        # Now decode z.
        Y : torch.Tensor    = self.Decode(Z);

        # All done! Hopefully Y \approx X.
        return Y;



    def latent_initial_conditions(  self,
                                    param_grid     : numpy.ndarray, 
                                    physics        : Physics) -> list[list[numpy.ndarray]]:
        """
        This function maps a set of initial conditions for the FOM to initial conditions for the 
        latent space dynamics. Specifically, we take in a set of possible parameter values. For 
        each set of parameter values, we recover the FOM IC (from physics), then map this FOM IC to 
        a latent space IC (by encoding it). We do this for each parameter combination and then 
        return a list housing the latent space ICs.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param_grid : numpy.ndarray, shape = (n_param, n_p)
            i,j element of this array holds the value of the j'th parameter in the i'th combination of 
            parameters. Here, n_param is the number of combinations of parameter values and n_p is the 
            number of parameters (in each combination).

        physics : Physics
            A "Physics" object that, among other things, stores the IC for each combination of 
            parameter values. This physics object should have the same number of initial conditions as 
            self.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        Z0 : list[list[numpy.ndarray]], len = n_param
            An n_param element list whose i'th element is an n_IC element list whose j'th element 
            is an numpy.ndarray of shape (1, n_z) whose k'th element holds the k'th component of 
            the encoding of the initial condition for the j'th derivative of the latent dynamics 
            corresponding to the i'th combination of parameter values.
        
            If we let U0_i denote the FOM IC for the i'th set of parameters, then the i'th element of 
            the returned list is [self.encoder(U0_i)].
        """

        # Checks.
        assert(isinstance(param_grid, numpy.ndarray));
        assert(len(param_grid.shape) == 2);
        assert(physics.n_IC     == self.n_IC);

        # Figure out how many combinations of parameter values there are.
        n_param     : int                       = param_grid.shape[0];
        Z0          : list[list[numpy.ndarray]] = [];
        LOGGER.debug("Encoding initial conditions for %d parameter values" % n_param);

        # Cycle through the parameters.
        for i in range(n_param):
            # Fetch the IC for the i'th set of parameters. Then map it to a tensor.
            u0_np   : numpy.ndarray = physics.initial_condition(param_grid[i])[0];
            u0      : torch.Tensor  = torch.Tensor(u0_np).reshape((1,) + u0_np.shape);

            # Encode the IC, then map the encoding to a numpy array.
            z0      : numpy.ndarray = self.Encode(u0).detach().numpy();

            # Append the new IC to the list of latent ICs
            Z0.append([z0]);

        # Return the list of latent ICs.
        return Z0;



    def export(self) -> dict:
        """
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        This function extracts everything we need to recreate self from scratch. Specifically, we 
        extract the encoder/decoder state dictionaries, self's architecture, activation function 
        and reshape_shape. We store and return this information in a dictionary.
         
        You can pass the returned dictionary to the load_Autoencoder method to generate an 
        Autoencoder object that is identical to self.
        """

        # TO DO: deep export which includes all information needed to re-initialize self from 
        # scratch. This would probably require changing the initializer.

        dict_ = {   'encoder state'  : self.encoder.cpu().state_dict(),
                    'decoder state'  : self.decoder.cpu().state_dict(),
                    'widths'         : self.widths, 
                    'activations'    : self.activations, 
                    'reshape_shape'  : self.reshape_shape};
        return dict_;



def load_Autoencoder(dict_ : dict) -> Autoencoder:
    """
    This function builds an Autoencoder object using the information in dict_. dict_ should be 
    the dictionary returned by the export method for some Autoencoder object (or a de-serialized 
    version of one). The Autoencoder that we recreate should be an identical copy of the object 
    that generated dict_.

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    dict_: dict
        This should be a dictionary returned by an Autoencoder's export method.

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    AE : Autoencoder 
        An Autoencoder object that is identical to the one that created dict_!
    """

    LOGGER.info("De-serializing an Autoencoder..." );

    # First, extract the parameters we need to initialize an Autoencoder object with the same 
    # architecture as the one that created dict_.
    widths          : list[int] = dict_['widths'];
    activations     : list[str] = dict_['activations'];
    reshape_shape   : list[int] = dict_['reshape_shape'];

    # Now... initialize an Autoencoder object.
    AE = Autoencoder(widths = widths, activations = activations, reshape_shape = reshape_shape);

    # Now, update the encoder/decoder parameters.
    AE.encoder.load_state_dict(dict_['encoder state']); 
    AE.decoder.load_state_dict(dict_['decoder state']); 

    # All done, AE is now identical to the Autoencoder object that created dict_.
    return AE;

