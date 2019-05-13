import mxnet as mx
import numpy as np
from sklearn import preprocessing
class Model(object):
    def __init__(self,
                 ctx=mx.cpu(),
                 image_size=[112,112],
                 model_str="../models/model-r100-ii/model, 0"
                 ):

        layer="fc1"
        _vec = model_str.split(',')
        assert len(_vec) == 2
        prefix = _vec[0]
        epoch = int(_vec[1])
        print('loading', prefix, epoch)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers[layer + '_output']
        self.model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        # model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
        self.model.bind(data_shapes=[('data', (10, 3, image_size[0], image_size[1]))])
        self.model.set_params(arg_params, aux_params)


    def get_vector(self,points):
        input_blob = np.expand_dims(points, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = preprocessing.normalize(embedding).flatten()
        return embedding