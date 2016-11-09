import caffe
import numpy as np
import yaml
from helpers import *

class MyLayer(caffe.Layer):

    def parse_file(self, file_name):
        f= open(fileName)
        lines = [line.rstrip('\n') for line in f]
        imageList=[]

        for i in lines:
            temp= i.split(' ')
            imageList.append( ( temp[0], int(temp[1]) )
   
    def _shuffle_ids(self):
        self._perm = np.random.permutation( np.arange( self.length_files))
        self._cur = 0
    
    def _get_next_m_batch_ids(self):
        if self._cur + 2*self.batch_size >= self.length_files:
            self._shuffle_ids()
        m_batch_ids1 = self._perm[ self._cur: self._cur + self.batch_size]
        self._cur += self.batch_size
        m_batch_ids2 = self._perm[ self._cur: self._cur + self.batch_size]
        self._cur += self.batch_size
        return m_batch_ids1, m_batch_ids2
        

    def _get_next_m_batch(self):
        #TODO use prefetch option
        m_batch_ids1, m_batch_ids2 = self._get_next_m_batch_ids()
        
        #TODO load each image list form helper function
        #write method to calculate similarity labels
        blob1= helper._get_image_list_blob( m_batch_ids1, self.mean_image)
        #TODO get from blob 2
        #TODO construct similarity blob


    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        #self.num = layer_params["num_classes"]
        #print "Parameter num : ", self.num
        self.source_file = layer_params["file_name"]
        self.image_list = parse_file(self.source_file)
        self.length_files = len(self.image_file)
        #TODO load mean file
        self.batch_size = layer_params["batch_size"]
        self._name_to_top= {
          'data': 0,
          'data_p': 1,
          'sim': 2}
        top[0].reshape(1,3,227,227)
        top[1].reshape(1,3,227,227)
        top[2].reshape(1)

        self.mean_image = helper._get_image_from_binaryproto( mean_file_name)


    def reshape(self, bottom, top):
        #if using batch mode do reshaping when calling forward
        pass

    def forward(self, bottom, top):
        
        blobs= self._get_next_m_batch()
        #top[0].reshape(*bottom[0].shape)
        #top[0].data[...] = bottom[0].data + self.num
        #TODO from blob put data in top

    def backward(self, top, propagate_down, bottom):
        pass
