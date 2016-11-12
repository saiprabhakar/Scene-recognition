import caffe
import numpy as np
import yaml
from helpers import _get_image_from_binaryproto, _get_image_list_blob, _get_sim_list_blob


class MyLayer(caffe.Layer):
    def _parse_file(self, file_name, image_source):
        f = open(file_name)
        lines = [line.rstrip('\n') for line in f]
        imageList = []
        #import pdb
        #pdb.set_trace()
        for i in lines:
            temp = i.split(' ')
            imageList.append((image_source + temp[0], int(temp[1])))
        return imageList

    def _shuffle_ids(self):
        self._perm = np.random.permutation(np.arange(self.length_files))
        self._cur = 0

    def _get_next_m_batch_ids(self):
        if self._cur + 2 * self.batch_size >= self.length_files:
            self._shuffle_ids()
        m_batch_ids1 = self._perm[self._cur:self._cur + self.batch_size]
        self._cur += self.batch_size
        m_batch_ids2 = self._perm[self._cur:self._cur + self.batch_size]
        self._cur += self.batch_size
        return m_batch_ids1, m_batch_ids2

    def _get_next_m_batch(self):
        #TODO use prefetch option
        m_batch_ids1, m_batch_ids2 = self._get_next_m_batch_ids()

        self.m_batch_1 = [self.image_list[i] for i in m_batch_ids1]
        self.m_batch_2 = [self.image_list[i] for i in m_batch_ids2]
        blob1 = _get_image_list_blob(self.m_batch_1, self.mean_image,
                                     self.scale_min_size,
                                     self.final_image_size)
        blob2 = _get_image_list_blob(self.m_batch_2, self.mean_image,
                                     self.scale_min_size,
                                     self.final_image_size)
        blobSim = _get_sim_list_blob(self.m_batch_1, self.m_batch_2)

        blobs = {"data": blob1, "data_p": blob2, "sim": blobSim}
        #TODO verify blob value
        return blobs

    def setup(self, bottom, top):
        #import pdb
        #print "\n\n\nsetting python data layer"
        #pdb.set_trace()
        layer_params = yaml.load(self.param_str)
        #source_params = yaml.load(self.image_str)
        #mean_params = yaml.load(self.mean_str)

        self.source_file = layer_params["file_name"]
        self.image_source = layer_params["image_source"]
        self.image_list = self._parse_file(self.source_file, self.image_source)
        self.length_files = len(self.image_list)
        self._cur = 0
        self.batch_size = layer_params["batch_size"]
        self._shuffle_ids()

        self.final_image_size = layer_params["final_image_size"]
        self.scale_min_size = layer_params["scale_min_size"]
        self.num_channels = layer_params["num_channels"]

        self.mean_image = _get_image_from_binaryproto(layer_params[
            "mean_image"])
        assert self.mean_image.shape[1] == self.scale_min_size
        assert self.mean_image.shape[0] == self.scale_min_size
        assert self.mean_image.shape[2] == self.num_channels

        self._name_to_top = {'data': 0, 'data_p': 1, 'sim': 2}
        top[0].reshape(2, self.num_channels, self.final_image_size,
                       self.final_image_size)
        top[1].reshape(2, self.num_channels, self.final_image_size,
                       self.final_image_size)
        top[2].reshape(2)

    def reshape(self, bottom, top):
        #if using batch mode do reshaping when calling forward
        pass

    def forward(self, bottom, top):
        print "in forward"
        blobs = self._get_next_m_batch()
        #import IPython
        #IPython.embed()
        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top[blob_name]
            top[top_ind].reshape(*(blob.shape))
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)
            #print "top ", blob_name, top[top_ind].data.shape
        self.tp = top
        #import IPython
        #IPython.embed()
        print "out forward"

    def backward(self, top, propagate_down, bottom):
        pass
