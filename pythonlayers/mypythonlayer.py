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
        #replace number with variable in the below fn
        blob1= _get_image_list_blob( m_batch_ids1, self.mean_image, self.scale_min_size, self.final_image_size )
        blob2= _get_image_list_blob( m_batch_ids2, self.mean_image, self.scale_min_size, self.final_image_size )
        #TODO get from blob 2
        #TODO construct similarity blob


    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        
        self.source_file = layer_params["file_name"]
        self.image_list = parse_file(self.source_file)
        self.length_files = len(self.image_file)
        
        self.batch_size = layer_params["batch_size"]
        

        self.final_image_size = layer_params["final_image_size"]
        self.scale_min_size = layer_params["scale_min_size"]
        self.num_channels = layer_params["num_channels"]
        
        self.mean_image = _get_image_from_binaryproto( layer_params["mean_image"])
        assert self.mean_image.shape[1] == self.scale_min_size 
        assert self.mean_image.shape[0] == self.scale_min_size 
        assert self.mean_image.shape[2] == self.num_channels 
        
        self._name_to_top= {
          'data': 0,
          'data_p': 1,
          'sim': 2}
        top[0].reshape(1, self.num_channels, self.final_image_size, self.final_image_size)
        top[1].reshape(1, self.num_channels, self.final_image_size, self.final_image_size)
        top[2].reshape(1)


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
