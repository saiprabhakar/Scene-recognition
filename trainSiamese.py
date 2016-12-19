# --------------------------------------------------------
# floor_recog
# Written by Sai Prabhakar
# CMU-RI Masters
# --------------------------------------------------------

from siamese.SiameseTrainer import *

pretrained_model_proto = 'placesOriginalModel/places_processed.prototxt'
pretrained_model = 'placesOriginalModel/places205CNN_iter_300000_upgraded.caffemodel'
netSize = 100
siameseSolver = 'siameseModels/siamesePlaces_' + str(
    netSize) + '_solver.prototxt'
fileName = 'data/imagelist.txt'
train = 0
#pretrainedSiameseModel = 'results/-netsize-1000-epoch-65-tstamp--Timestamp-2016-12-18-16:17:08-net.caffemodel'
#pretrainedSiameseModel = 'results/-netsize-100-epoch-65-tstamp--Timestamp-2016-12-18-16:16:30-net.caffemodel'
pretrainedSiameseModel = None
testProto = 'siameseModels/siamesePlaces_' + str(netSize) + '_test.prototxt'

siameseTrainer(
    siameseSolver=siameseSolver,
    pretrainedSiameseModel=pretrainedSiameseModel,
    fileName=fileName,
    pretrained_model=pretrained_model,
    pretrained_model_proto=pretrained_model_proto,
    testProto=testProto,
    train=train,
    netSize=netSize)
