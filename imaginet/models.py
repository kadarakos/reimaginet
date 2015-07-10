from funktional.layer import Layer, Dense, StackedGRU, StackedGRUH0, \
                             Embedding, OneHot, \
                             last, softmax3d
import funktional.util as util
from funktional.util import CosineDistance, CrossEntropy, linear, clipped_rectify
from funktional.util import autoassign, params
import funktional.context as context
import theano.tensor as T
import theano

class Activation(Layer):
    """Activation function object."""
    def __init__(self, activation):
        autoassign(locals())
        self.params = []

    def __call__(self, inp):
        return self.activation(inp)

class Visual(Layer):
    def __init__(self, size_embed, size, size_out, depth,
                 gru_activation=clipped_rectify,
                 visual_activation=linear):
        autoassign(locals())
        self.EncodeV = StackedGRUH0(self.size_embed, self.size, self.depth,
                                     activation=self.gru_activation)
        self.ToImg   = Dense(self.size, self.size_out)
        self.params  = params(self.EncodeV, self.ToImg)

    def __call__(self, inp):
        return self.visual_activation(self.ToImg(last(self.EncodeV(inp))))
            
class MultitaskLM(Layer):
    """Visual encoder combined with a textual task."""
    
    def __init__(self, size_vocab, size_embed, size, size_out, depth,
                 gru_activation=clipped_rectify,
                 visual_activation=linear):
        autoassign(locals())
        self.Embed   =  Embedding(self.size_vocab, self.size_embed)
        self.Visual  =  Visual(self.size_embed, self.size, self.size_out, self.depth,
                               gru_activation=self.gru_activation,
                               visual_activation=self.visual_activation)
        self.LM      =  StackedGRUH0(self.size_embed, self.size, self.depth,
                                     activation=self.gru_activation)
        self.ToTxt   =  Dense(self.size, self.size_embed) # try direct softmax
        self.params  =  params(self.Embed, self.Visual, self.LM, self.ToTxt)

        
    def __call__(self, inp, output_prev, _img):
        img_pred   = self.Visual(self.Embed(inp))
        txt_pred   = softmax3d(self.Embed.unembed(self.ToTxt(self.LM(self.Embed(output_prev)))))
        return (img_pred, txt_pred)

    
    def predictor_v(self):
        """Return function to predict image vector from input."""
        input    = T.imatrix()
        return theano.function([input],
                         self.Visual(self.Embed(input)))

class MultitaskLMC(Layer):
    """Visual encoder combined with a textual decoder."""
    
    def __init__(self, size_vocab, size_embed, size, size_out, depth,
                 gru_activation=clipped_rectify,
                 visual_activation=linear):
        autoassign(locals())
        self.Embed   =  Embedding(self.size_vocab, self.size_embed)
        self.Visual  =  Visual(self.size_embed, self.size, self.size_out, self.depth,
                               gru_activation=self.gru_activation,
                               visual_activation=self.visual_activation)
        self.LM      =  StackedGRU(self.size_embed, self.size, self.depth,
                                     activation=self.gru_activation)
        self.FromImg =  Dense(self.size_out, self.size)
        self.ToTxt   =  Dense(self.size, self.size_embed) # try direct softmax
        self.params  =  params(self.Embed, self.Visual, self.LM, self.FromImg, self.ToTxt)

        
    def __call__(self, inp, output_prev, img):
        img_pred   = self.Visual(self.Embed(inp))
        txt_pred   = softmax3d(self.Embed.unembed(self.ToTxt(self.LM(self.FromImg(img),
                                                                     self.Embed(output_prev)))))
        return (img_pred, txt_pred)

    
    def predictor_v(self):
        """Return function to predict image vector from input."""
        input    = T.imatrix()
        return theano.function([input],
                         self.Visual(self.Embed(input)))
    
class Imaginet(object):
    """Trainable imaginet model."""

    def __init__(self, size_vocab, size_embed, size, size_out, depth, network, alpha=0.5,
                 gru_activation=clipped_rectify, visual_activation=linear, cost_visual=CosineDistance,
                 max_norm=None):
        autoassign(locals())
        self.network = network(self.size_vocab, self.size_embed, self.size, self.size_out, self.depth,                                gru_activation=self.gru_activation, visual_activation=self.visual_activation)
                               
        input         = T.imatrix()
        output_t_prev = T.imatrix()
        output_t      = T.imatrix()
        output_v      = T.fmatrix()
        self.OH       = OneHot(size_in=self.size_vocab)
        output_t_oh   = self.OH(output_t)
        # TRAINING
        with context.context(training=True):
            output_v_pred, output_t_pred = self.network(input, output_t_prev, output_v)
            cost_T = CrossEntropy(output_t_oh, output_t_pred)
            cost_V = self.cost_visual(output_v, output_v_pred)
            cost = self.alpha * cost_T + (1.0 - self.alpha) * cost_V
        #TESTING
        with context.context(training=False):
            output_v_pred_test, output_t_pred_test = self.network(input, output_t_prev, output_v)
            cost_T_test = CrossEntropy(output_t_oh, output_t_pred_test)
            cost_V_test = self.cost_visual(output_v, output_v_pred_test)
            cost_test = self.alpha * cost_T_test + (1.0 - self.alpha) * cost_V_test
        self.updater = util.Adam(max_norm=self.max_norm)
        updates = self.updater.get_updates(self.network.params, cost)
        # TODO better way of dealing with needed/unneeded output_t_prev?
        self.train = theano.function([input, output_v, output_t_prev, output_t ], [cost, cost_T, cost_V],
                                     updates=updates, on_unused_input='warn')

        self.loss_test = theano.function([input, output_v, output_t_prev, output_t ],
                                         [cost_test, cost_T_test, cost_V_test],
                                    on_unused_input='warn')


# Functions added outside the class do not interfere with loading of older versions
def predictor_v(model):
    """Return function to predict image vector from input using `model`."""
    return model.network.predictor_v()

    
