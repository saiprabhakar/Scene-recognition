#/usr/bin/python
# --------------------------------------------------------
# floor_recog
# Written by Sai Prabhakar
# CMU-RI Masters
# --------------------------------------------------------

import sys
import os
os.environ['GLOG_minloglevel'] = '3'

import argparse
import numpy as np
import cv2
import caffe
from pythonlayers.helpers import im_to_blob
import matplotlib.pyplot as plt
import datetime

im_target_size = 227

blob = caffe.proto.caffe_pb2.BlobProto()
data = open('placesOriginalModel/places205CNN_mean.binaryproto', 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
# changing order of rows, colmn, channel, batch_size
arr = np.squeeze(arr.transpose((2, 3, 1, 0)))
im_scaley = float(im_target_size) / float(256)
im_scalex = float(im_target_size) / float(256)
meanarr = cv2.resize(
    arr,
    None,
    None,
    fx=im_scalex,
    fy=im_scaley,
    interpolation=cv2.INTER_LINEAR)


def _round_image(img):
    img[img > 255] = 255
    img[img < 0] = 0
    return img


def _get_image_blob(img_name):

    im = _load_image(img_name)
    #im = cv2.imread(img_name)
    #print im.shape, meanarr.shape
    processed_ims = im - meanarr
    #processed_ims = im
    #IPython.embed()
    blob = im_to_blob(processed_ims)

    return blob


def _load_image(img_name):
    #TODO crop the image to a square one then reshape
    im = cv2.imread(img_name)
    im_orig = im.astype(np.float32, copy=True)
    #im_shape = im_orig.shape
    #im_size = im_shape[0:2]  #rows, colmns y,x

    min_curr_size = min(im.shape[:2])
    im_scale = float(im_target_size) / float(min_curr_size)

    #im_scaley = float(im_target_size) / float(im_size[0])
    #im_scalex = float(im_target_size) / float(im_size[1])
    im = cv2.resize(
        im_orig[:min_curr_size, :min_curr_size, :],
        None,
        None,
        fx=im_scale,
        fy=im_scale,
        interpolation=cv2.INTER_LINEAR)
    return im.copy()


def _get_occluded_image_blobs(img_name, size_patch, stride):
    im = _load_image(img_name)

    cR = 0
    l_blob = []
    l_occ_map = []
    while im_target_size - 1 > cR + size_patch - 1:
        cC = 0
        while im_target_size - 1 > cC + size_patch - 1:
            #import IPython
            #IPython.embed()

            occluded_image, occ_map = _occlude_image(im.copy(), cR, cC,
                                                     size_patch, stride)
            cC += stride

            #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            #cv2.imshow('image',occluded_image.astype(np.uint8))
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            processed_ims = occluded_image - meanarr
            #IPython.embed()
            l_blob.append(im_to_blob(processed_ims))
            l_occ_map.append(occ_map)
        cR += stride

    return l_blob, l_occ_map


def _occlude_image(im, cR, cC, size_patch, stride):
    """creates gray patches in image."""
    im[cR:cR + stride, cC:cC + stride, :] = 127.5
    occ_map = np.ones((im_target_size, im_target_size))
    occ_map[cR:cR + stride, cC:cC + stride] = 0
    #import IPython
    #IPython.embed()
    return im, occ_map


class SiameseTrainWrapper2(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self,
                 solver_prototxt,
                 pretrainedSiameseModel=None,
                 pretrained_model=None,
                 pretrained_model_proto=None,
                 testProto=None,
                 train=1,
                 netSize=1000,
                 class_size=6,
                 class_adju=2):
        """Initialize the SolverWrapper."""
        caffe.set_device(0)
        caffe.set_mode_gpu()
        #caffe.set_mode_cpu()
        self.train = train
        self.netSize = netSize
        self.class_size = class_size
        self.class_adju = class_adju
        if self.train == 1:
            self.solver = caffe.SGDSolver(solver_prototxt)
            if pretrainedSiameseModel is not None:
                print('Loading pretrained model '
                      'weights from {:s}').format(pretrainedSiameseModel)
                self.solver.net.copy_from(pretrainedSiameseModel)

            elif pretrained_model is not None:
                self.solver.net.copy_from(pretrained_model)

            else:
                print('Initializing completely from scratch .... really ?')

            self.solver.test_nets[0].share_with(self.solver.net)
        else:
            self.siameseTestNet = caffe.Net(testProto, pretrainedSiameseModel,
                                            caffe.TEST)

    def trainTest(self):
        #import ipdb
        #ipdb.set_trace()
        #self.solver.test_nets[0].forward()
        #self.solver.net.forward()
        #self.solver.test_nets[0].blobs['conv1'].data[0,0,1,1:5]
        #self.solver.net.blobs['conv1'].data[0,0,1,1:5]
        #import IPython
        #IPython.embed()

        #print self.solver.net.params['conv1'][0].data[1,1,1:5,1]
        #print self.solver.test_nets[0].params['conv1'][0].data[1,1,1:5,1]
        #1500
        num_data_epoch_train = 540
        num_data_epoch_test = 240
        tStamp = '-Timestamp-{:%Y-%m-%d-%H:%M:%S}'.format(
            datetime.datetime.now())
        plt.ion()
        try:
            for k in range(100):
                disLoss = 0
                simLoss = 0
                simC = 0
                disC = 0
                plot_data_d = np.zeros((0, 2))
                plot_data_s = np.zeros((0, 2))
                plot_data_id_l1 = np.zeros((0, 2))
                plot_data_id_l2 = np.zeros((0, 2))
                lossId1s = 0
                lossId2s = 0
                for i in range(num_data_epoch_train):
                    self.solver.step(1)

                    #if i%50 == 0 and i != 0:
                    #import IPython
                    #IPython.embed()

                    lossCo = 0.2 * self.solver.net.blobs['cont_loss'].data
                    lossId1 = 0.4 * self.solver.net.blobs[
                        'softmax_loss_1'].data
                    lossId2 = 0.4 * self.solver.net.blobs[
                        'softmax_loss_2'].data
                    if self.solver.net.blobs['sim'].data == 1:
                        simC += 1
                        simLoss += lossCo
                        plot_data_s = np.vstack(
                            (plot_data_s, [k + 0.5, lossCo]))
                    else:
                        disC += 1
                        disLoss += lossCo
                        plot_data_d = np.vstack((plot_data_d, [k, lossCo]))
                    lossId1s += lossId1
                    lossId2s += lossId2

                    #print "sim", self.solver.net.blobs[
                    #    'sim'].data, "cont loss", lossCo, "id1", lossId1, "id2", lossId2
                plot_data_id_l1 = np.vstack(
                    (plot_data_id_l1, [k, lossId1s / num_data_epoch_train]))
                plot_data_id_l2 = np.vstack(
                    (plot_data_id_l2, [k, lossId2s / num_data_epoch_train]))
                print k, "cont net loss", simLoss / (simC + 0.1), disLoss / (
                    disC + 0.1), simC, disC, "Id net loss", lossId1s, lossId2s
                plt.figure(1)
                plt.xlim(-0.5, 100)
                plt.title(str(self.netSize) + "train errors")
                plt.plot(plot_data_s[:, 0], plot_data_s[:, 1], 'r.')
                plt.plot(plot_data_d[:, 0], plot_data_d[:, 1], 'b.')
                plt.pause(0.05)

                plt.figure(2)
                plt.xlim(-0.5, 100)
                plt.title(str(self.netSize) + "train softmax loss")
                plt.plot(plot_data_id_l1[:, 0], plot_data_id_l1[:, 1], 'r.')
                plt.plot(plot_data_id_l2[:, 0], plot_data_id_l2[:, 1], 'b.')
                plt.pause(0.05)

                if k % 5 == 0:
                    acc1 = np.zeros(self.class_size)
                    acc2 = np.zeros(self.class_size)
                    fre1 = np.zeros(self.class_size)
                    fre2 = np.zeros(self.class_size)
                    plot_acc1 = np.zeros((0, 2))
                    plot_acc2 = np.zeros((0, 2))
                    confusion_dis = np.zeros(
                        (self.class_size, self.class_size))
                    for i in range(num_data_epoch_test):
                        loss1 = self.solver.test_nets[0].forward()
                        #print i, loss1, loss1['sim'], loss1['euc_dist']
                        #import IPython
                        #IPython.embed()

                        if loss1['sim'] == 1:
                            simC += 1
                            simLoss += loss1['euc_dist']
                            plot_data_s = np.vstack(
                                (plot_data_s, [k + 0.5, loss1['euc_dist']]))
                        else:
                            disC += 1
                            disLoss += loss1['euc_dist']
                            plot_data_d = np.vstack(
                                (plot_data_d, [k, loss1['euc_dist']]))
                        id1 = self.solver.test_nets[0].layers[0].m_batch_1[0][
                            1] - self.class_adju
                        id2 = self.solver.test_nets[0].layers[0].m_batch_2[0][
                            1] - self.class_adju
                        confusion_dis[id1, id2] += loss1['euc_dist']
                        acc1[int(self.solver.test_nets[0].blobs['label1'].data[
                            0])] += self.solver.test_nets[0].blobs[
                                'accuracy1'].data
                        fre1[int(self.solver.test_nets[0].blobs['label1'].data[
                            0])] += 1
                        acc2[int(self.solver.test_nets[0].blobs['label2'].data[
                            0])] += self.solver.test_nets[0].blobs[
                                'accuracy2'].data
                        fre2[int(self.solver.test_nets[0].blobs['label2'].data[
                            0])] += 1
                    netacc = (acc1.sum() + acc2.sum()) / (
                        fre1.sum() + fre2.sum())
                    netacc1 = (acc1.sum()) / (fre1.sum())
                    netacc2 = (acc2.sum()) / (fre2.sum())
                    acc1 = acc1 / (fre1 + 0.1)
                    acc2 = acc2 / (fre2 + 0.1)
                    plot_acc1 = np.vstack((plot_acc1, [k, netacc1]))
                    plot_acc2 = np.vstack((plot_acc2, [k, netacc2]))
                    print "testing**** net loss", simLoss / (
                        simC + 0.1), disLoss / (disC + 0.1), simC, disC
                    #print confusion_dis
                    print acc1, acc2
                    print netacc
                    plt.figure(3)
                    plt.xlim(-0.5, 100)
                    plt.title(str(self.netSize) + "test accuracy")
                    plt.plot(k, netacc1, 'r.')
                    plt.plot(k, netacc2, 'b.')
                    plt.pause(0.05)
                    #import IPython
                    #IPython.embed()

                    plt.figure(4)
                    #plt.clf()
                    #plt.xlim(-0.5, 1.5)
                    plt.xlim(-0.5, 100)
                    plt.title(str(self.netSize) + "test distance")
                    plt.plot(plot_data_s[:, 0], plot_data_s[:, 1], 'r.')
                    plt.plot(plot_data_d[:, 0], plot_data_d[:, 1], 'b.')
                    #plt.show()
                    plt.pause(0.05)

                if k % 1 == 0:
                    preName = 'modifiedNetResults/' + 'Modified-netsize-' + str(
                        self.netSize) + '-epoch-' + str(
                            k) + '-tstamp-' + tStamp
                    self.solver.net.save(preName + '-net.caffemodel')

        except KeyboardInterrupt:
            pass

        preName = 'modifiedNetResults/' + 'Modified-netsize-' + str(
            self.netSize) + '-epoch-' + str(k) + '-tstamp-' + tStamp
        plt.ioff()

        plt.figure(1).savefig(preName + '-train-d-error.png')
        plt.figure(1).savefig(preName + '-train-s-error.png')
        plt.figure(3).savefig(preName + '-test-acc.png')
        plt.figure(4).savefig(preName + '-test-dist.png')
        self.solver.net.save(preName + '-net-final.caffemodel')
        plt.close('all')

    def visualizing_m1(self, fileName):
        tStamp = '-Timestamp-{:%Y-%m-%d-%H:%M:%S}'.format(
            datetime.datetime.now())

        f = open(fileName)
        lines = [line.rstrip('\n') for line in f]
        imageDict = {}
        imlist = []

        size_patch = 30
        stride = 15
        #TODO adaptive thresholding
        #TODO padded occlusion
        threshold = 50

        for i in lines:
            temp = i.split(' ')
            imageDict[temp[0]] = int(temp[1]) - 2
            imlist.append(temp[0])
        for i in range(len(imlist)):
            im1 = i
            #occlude im1 and get map on im1
            print 'generating heat map for ', imageDict[imlist[im1]], imlist[
                im1]
            im_gen = self.generate_heat_map_softmax(
                imageDict, imlist, im1, size_patch, stride, threshold)
            preName = 'modifiedNetResults_visu/' + imlist[im1] + '-' + str(
                size_patch) + str(stride) + '-' + '-M-nSize-' + str(
                    self.netSize) + '-tstamp-' + tStamp
            cv2.imwrite(preName + '.png', im_gen)

    def generate_heat_map_softmax(self,
                                  imageDict,
                                  imlist,
                                  im1,
                                  size_patch,
                                  stride,
                                  threshold=60):
        #size_patch = 30
        #stride = 15
        offset = 200
        l_blobs_im1, l_occ_map = _get_occluded_image_blobs(
            img_name='data/' + imlist[im1],
            size_patch=size_patch,
            stride=stride)
        print "no of occluded maps ", len(l_blobs_im1)
        #blob_im2 = _get_image_blob('data/' + imlist[im2])
        blobs = {'data': None}
        #blobs['data'] = blob_im2
        heat_map = np.zeros((im_target_size, im_target_size))

        for i in range(len(l_blobs_im1)):
            blobs['data'] = l_blobs_im1[i]

            blobs_out1 = self.siameseTestNet.forward(data=blobs['data'].astype(
                np.float32, copy=True))

            #import IPython
            #IPython.embed()

            p = self.siameseTestNet.blobs['fc9_f'].data[0]
            p = p - p.min()
            p = p / p.sum()
            prob1 = p[imageDict[imlist[im1]]]
            #prob1 = blobs_out1['prob1'][0][imageDict[imlist[im1]]]
            #print "prob1 ", prob1
            heat_map += l_occ_map[i] * prob1
            #import IPython
            #IPython.embed()

        #import IPython
        #IPython.embed()

        heat_map = heat_map[:offset, :offset]
        #heat_map = np.log(heat_map + 1)

        heat_map_o = 100 * (heat_map - heat_map.min()) / (
            heat_map.max() - heat_map.min())
        img1 = _load_image('data/' + imlist[im1])
        img1 = img1[:offset, :offset, :]
        #img2 = _load_image('data/' + imlist[im2])

        #img1_heat = img1.copy()
        #img1_heat[:, :, 2] = heat_map
        heat_map = heat_map_o.copy()

        #invert the heat map
        heat_map = 100 - heat_map
        heat_map[heat_map < threshold] = 0

        heat_map *= 1.5
        #plt.matshow(heat_map)
        #plt.colorbar()
        #plt.show()

        img1_o = img1.copy()
        temp = np.sum(img1, axis=2) / 3.0
        img1[:, :, 2] = temp
        img1[:, :, 1] = temp
        img1[:, :, 0] = temp

        img1[:, :, 2] += heat_map

        #img1[:, :, 1] -= heat_map
        #img1[:, :, 0] -= heat_map
        #cv2.imshow('image', img1.astype(np.uint8))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        #import IPython
        #IPython.embed()
        #import IPython
        #IPython.embed()
        #fig0 = plt.figure()
        #f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        #ax1.imshow(img1_heat)
        #ax1.set_title('Sharing Y axis')
        #ax2.imshow(img2)
        #plt.show()
        img1 = _round_image(img1)
        return img1.astype(np.uint8)


def siameseTrainer(siameseSolver,
                   fileName,
                   pretrained_model,
                   pretrainedSiameseModel,
                   testProto,
                   pretrained_model_proto,
                   train=1,
                   visu=0,
                   netSize=1000):
    #numImagePair = 1  #len(imdb.image_index)
    # timers
    #_t = {'im_detect' : Timer(), 'misc' : Timer()}

    sw = SiameseTrainWrapper2(
        siameseSolver,
        pretrainedSiameseModel=pretrainedSiameseModel,
        pretrained_model=pretrained_model,
        pretrained_model_proto=pretrained_model_proto,
        testProto=testProto,
        train=train,
        netSize=netSize)
    # import IPython
    # IPython.embed()

    if train == 1:
        print "training"
        sw.trainTest()
    elif visu == 0:
        print "testing with ", pretrainedSiameseModel
        sw.test(fileName)
    else:
        print 'visalizing with ', pretrainedSiameseModel
        sw.visualizing_m1(fileName)

    #sw = SiameseWrapper(siameseSolver, pretrained_model=pretrained_model, pretrained_model_proto= pretrained_model_proto, train=0 )

    #sw.testCode( fileName)
