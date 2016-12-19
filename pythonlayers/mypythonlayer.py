# --------------------------------------------------------
# floor_recog
# Written by Sai Prabhakar
# CMU-RI Masters
# --------------------------------------------------------

#coustom python data layer

import caffe
import numpy as np
import yaml
from helpers import _get_image_from_binaryproto, _get_image_list_blob, _get_sim_list_blob
from random import randint


class MyLayer(caffe.Layer):
    def _parse_file(self, file_name, image_source):
        """Loads file_names and the class labels
        and generates pairs of images
        for siamese training.
        """
        f = open(file_name)
        lines = [line.rstrip('\n') for line in f]
        imageList = []
        self.m_img = 0
        self.n_img = 0
        self.m_pairs = []
        self.n_pairs = []

        #import pdb
        #pdb.set_trace()
        for i in lines:
            temp = i.split(' ')
            imageList.append((image_source + temp[0], int(temp[1])))
        self.image_list = imageList
        self.length_files = len(self.image_list)

        #creating combinations of images
        import itertools
        all_pairs = list(
            itertools.product(
                range(self.length_files), range(self.length_files)))
        for i in range(len(all_pairs)):
            if self.image_list[all_pairs[i][0]][1] == self.image_list[
                    all_pairs[i][1]][1]:
                self.m_pairs.append(all_pairs[i])
                self.m_img += 1
            else:
                self.n_pairs.append(all_pairs[i])
                self.n_img += 1

        #return imageList

        # def _shuffle_ids(self):
        #     self._perm = np.random.permutation(np.arange(self.length_files))
        #     self._cur = 0

    def _get_corrected_pairs(self):
        """Makes the number of positive and negative samples equal
        """
        #import IPython
        #IPython.embed()
        # making the lengths of similar and dissimilar images equal
        m_pairs = self.m_pairs
        n_pairs = self.n_pairs
        if len(m_pairs) >= len(n_pairs):
            #ind = np.random.permutation(len(m_pairs) - len(n_pairs))
            ind = len(m_pairs) - len(n_pairs)
            for i in range(ind):
                n_pairs.append(self.n_pairs[randint(0, len(self.n_pairs) - 1)])
        else:
            ind = len(n_pairs) - len(m_pairs)
            for i in range(ind):
                m_pairs.append(self.m_pairs[randint(0, len(self.m_pairs) - 1)])
        return m_pairs, n_pairs

    def _shuffle_pair_ids(self):
        """Makes the positive and negative samples equal.

        Shuffles the data pairs.
        """
        # print "shufle called"
        m_c_pairs, n_c_pairs = self._get_corrected_pairs()
        self._all_m_pairs = m_c_pairs + n_c_pairs
        print "total images in dataset ", len(self._all_m_pairs)
        self._perm = np.random.permutation(np.arange(len(self._all_m_pairs)))
        self._cur = 0

    def _get_next_m_batch_ids_pair(self):
        """Creates next mini batch file ids, depending on batch size.
        """
        if self._cur + self.batch_size >= len(self._all_m_pairs):
            # print self._cur, self.batch_size, len(self._all_m_pairs)
            self._shuffle_pair_ids()
        perm_ind = self._perm[self._cur:self._cur + self.batch_size]
        m_batch_ids1 = [self._all_m_pairs[i][0] for i in perm_ind]
        m_batch_ids2 = [self._all_m_pairs[i][1] for i in perm_ind]
        self._cur += self.batch_size
        return m_batch_ids1, m_batch_ids2

    # def _get_next_m_batch_ids(self):
    #     if self._cur + 2 * self.batch_size >= self.length_files:
    #         self._shuffle_ids()
    #     m_batch_ids1 = self._perm[self._cur:self._cur + self.batch_size]
    #     self._cur += self.batch_size
    #     m_batch_ids2 = self._perm[self._cur:self._cur + self.batch_size]
    #     self._cur += self.batch_size
    #     return m_batch_ids1, m_batch_ids2

    def _get_next_m_batch(self):
        """Creates data blobs for next mini batch.
        """
        #TODO use prefetch option
        m_batch_ids1, m_batch_ids2 = self._get_next_m_batch_ids_pair()

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
        return blobs

    def setup(self, bottom, top):
        """This function will be called by Caffe.

        Loads layer parameters and sets up data from source and
        data blobs.
        """
        #import pdb
        #print "\n\n\nsetting python data layer"
        #pdb.set_trace()
        layer_params = yaml.load(self.param_str)
        #source_params = yaml.load(self.image_str)
        #mean_params = yaml.load(self.mean_str)

        self.source_file = layer_params["file_name"]
        self.image_source = layer_params["image_source"]
        self._parse_file(self.source_file, self.image_source)
        self._cur = 0
        self.batch_size = layer_params["batch_size"]
        self._shuffle_pair_ids()

        self.final_image_size = layer_params["final_image_size"]
        self.scale_min_size = layer_params["scale_min_size"]
        self.num_channels = layer_params["num_channels"]

        self.mean_image = _get_image_from_binaryproto(layer_params[
            "mean_image"])
        assert self.mean_image.shape[1] == self.scale_min_size
        assert self.mean_image.shape[0] == self.scale_min_size
        assert self.mean_image.shape[2] == self.num_channels

        self._name_to_top = {'data': 0, 'data_p': 1, 'sim': 2}
        top[0].reshape(self.batch_size, self.num_channels,
                       self.final_image_size, self.final_image_size)
        top[1].reshape(self.batch_size, self.num_channels,
                       self.final_image_size, self.final_image_size)
        top[2].reshape(self.batch_size)

    def reshape(self, bottom, top):
        """This function will be called by Caffe.

        Reshaping based on batch size will be done at the
        during layer setup.

        Note: Contrastive loss dont support reshaping after
        layer setup.
        """
        pass

    def forward(self, bottom, top):
        """This function will be called by Caffe.

        Prepares data for next minibatcha and supplies them to
        corresponding top blobs.
        """
        #print "in forward"
        blobs = self._get_next_m_batch()
        #import IPython
        #IPython.embed()
        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top[blob_name]
            top[top_ind].reshape(*(blob.shape))
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)
            #print "top ", blob_name, top[top_ind].data.shape
        #self.tp = top
        #import IPython
        #IPython.embed()
        #print "out forward"

    def backward(self, top, propagate_down, bottom):
        """This function will be called by Caffe.

        Not necessary for this data layer.
        """
        pass
