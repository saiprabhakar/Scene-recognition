from siamese.SiameseTrainer import *


pretrained_model_proto= 'placesOriginalModel/places_processed.prototxt'
pretrained_model= 'placesOriginalModel/places205CNN_iter_300000_upgraded.caffemodel'
siameseSolver= 'siameseModels/siamesePlaces_solver.prototxt'
fileName= 'data/imagelist.txt'
siameseTrainer( siameseSolver=siameseSolver, fileName= fileName, pretrained_model=pretrained_model, pretrained_model_proto= pretrained_model_proto)
