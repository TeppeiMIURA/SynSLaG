# coding: utf-8
import os, sys
import argparse
import json
import cv2
import numpy as np
import tarfile
import shutil
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import scipy.io

J_PARENTS = {
    'Pelvis':'Pelvis', 'Spine1':'Pelvis', 'Spine2':'Spine1', 'Spine3':'Spine2', 'Neck':'Spine3', 'Head':'Neck',
    'L_Hip':'Pelvis', 'L_Knee':'L_Hip', 'L_Ankle':'L_Knee', 'L_Foot':'L_Ankle',
    'R_Hip':'Pelvis', 'R_Knee':'R_Hip', 'R_Ankle':'R_Knee', 'R_Foot':'R_Ankle',
    'L_Collar':'Spine3', 'L_Shoulder':'L_Collar', 'L_Elbow':'L_Shoulder', 'L_Wrist':'L_Elbow', 'L_Hand':'L_Wrist',
    'lthumb0':'L_Hand', 'lthumb1':'lthumb0', 'lthumb2':'lthumb1', 'lthumb_end':'lthumb2',
    'lindex0':'L_Hand', 'lindex1':'lindex0', 'lindex2':'lindex1', 'lindex_end':'lindex2',
    'lmiddle0':'L_Hand', 'lmiddle1':'lmiddle0', 'lmiddle2':'lmiddle1', 'lmiddle_end':'lmiddle2',
    'lring0':'L_Hand', 'lring1':'lring0', 'lring2':'lring1', 'lring_end':'lring2',
    'lpinky0':'L_Hand', 'lpinky1':'lpinky0', 'lpinky2':'lpinky1', 'lpinky_end':'lpinky2',
    'R_Collar':'Spine3', 'R_Shoulder':'R_Collar', 'R_Elbow':'R_Shoulder', 'R_Wrist':'R_Elbow', 'R_Hand':'R_Wrist',
    'rthumb0':'R_Hand', 'rthumb1':'rthumb0', 'rthumb2':'rthumb1', 'rthumb_end':'rthumb2',
    'rindex0':'R_Hand', 'rindex1':'rindex0', 'rindex2':'rindex1', 'rindex_end':'rindex2',
    'rmiddle0':'R_Hand', 'rmiddle1':'rmiddle0', 'rmiddle2':'rmiddle1', 'rmiddle_end':'rmiddle2',
    'rring0':'R_Hand', 'rring1':'rring0', 'rring2':'rring1', 'rring_end':'rring2',
    'rpinky0':'R_Hand', 'rpinky1':'rpinky0', 'rpinky2':'rpinky1', 'rpinky_end':'rpinky2',
}
J_COLOR = {
    'Pelvis':(255,0,0), 'Spine1':(255,0,0), 'Spine2':(255,0,0), 'Spine3':(255,0,0), 'Neck':(255,0,0), 'Head':(255,0,0),
    'L_Hip':(0,0,255), 'L_Knee':(0,0,255), 'L_Ankle':(0,0,255), 'L_Foot':(0,0,255),
    'R_Hip':(0,0,255), 'R_Knee':(0,0,255), 'R_Ankle':(0,0,255), 'R_Foot':(0,0,255),
    'L_Collar':(0,255,0), 'L_Shoulder':(0,255,0), 'L_Elbow':(0,255,0), 'L_Wrist':(0,255,0), 'L_Hand':(0,255,0),
    'lthumb0':(255,0,153), 'lthumb1':(255,0,153), 'lthumb2':(255,0,153), 'lthumb_end':(255,0,153),
    'lindex0':(204,153,51), 'lindex1':(204,153,51), 'lindex2':(204,153,51), 'lindex_end':(204,153,51),
    'lmiddle0':(102,0,204), 'lmiddle1':(102,0,204), 'lmiddle2':(102,0,204), 'lmiddle_end':(102,0,204),
    'lring0':(51,153,255), 'lring1':(51,153,255), 'lring2':(51,153,255), 'lring_end':(51,153,255),
    'lpinky0':(255,204,0), 'lpinky1':(255,204,0), 'lpinky2':(255,204,0), 'lpinky_end':(255,204,0),
    'R_Collar':(0,255,0), 'R_Shoulder':(0,255,0), 'R_Elbow':(0,255,0), 'R_Wrist':(0,255,0), 'R_Hand':(0,255,0),
    'rthumb0':(255,0,153), 'rthumb1':(255,0,153), 'rthumb2':(255,0,153), 'rthumb_end':(255,0,153),
    'rindex0':(204,153,51), 'rindex1':(204,153,51), 'rindex2':(204,153,51), 'rindex_end':(204,153,51),
    'rmiddle0':(102,0,204), 'rmiddle1':(102,0,204), 'rmiddle2':(102,0,204), 'rmiddle_end':(102,0,204),
    'rring0':(51,153,255), 'rring1':(51,153,255), 'rring2':(51,153,255), 'rring_end':(51,153,255),
    'rpinky0':(255,204,0), 'rpinky1':(255,204,0), 'rpinky2':(255,204,0), 'rpinky_end':(255,204,0),
}
def make_output_img(tmp_dir):
    tmp_j_2d_dir = tmp_dir + 'j2d/'
    os.makedirs(tmp_j_2d_dir, exist_ok=True)

    tmp_j_3d_dir = tmp_dir + 'j3d/'
    os.makedirs(tmp_j_3d_dir, exist_ok=True)

    with open(tmp_dir + 'keypoints.json', 'r') as f:
        json_dataset = json.load(f)

    tmp_img = cv2.imread(tmp_dir + json_dataset[0]['image'])
    h_img, w_img = tmp_img.shape[:2]

    for data in json_dataset:
        idx = data['image'].replace('.png', '').replace('image/img', '')
        j_2d = data['joints2D']
        j_3d = data['joints3D']

        # draw skeleton on 2d image
        j_2d_img = np.ones((h_img, w_img, 3), dtype=np.uint8) * 255

        for j in j_2d.keys():
            j_pos = (int(j_2d[j][0] * w_img),
                     int(j_2d[j][1] * h_img))
            parent_j_pos = (int(j_2d[J_PARENTS[j]][0] * w_img),
                            int(j_2d[J_PARENTS[j]][1] * h_img)) 
            
            j_2d_img = cv2.circle(j_2d_img, j_pos, 2, J_COLOR[j], -1)
            j_2d_img = cv2.line(j_2d_img, j_pos, parent_j_pos, J_COLOR[j], 1)

        cv2.imwrite(tmp_j_2d_dir + 'j2d' + idx + '.jpg', j_2d_img)

        # draw skeleton on 3D image
        fig = plt.figure()
        ax = Axes3D(fig)

        for j in j_3d.keys():
            j_pos = j_3d[j]
            parent_j_pos = j_3d[J_PARENTS[j]]

            c_rgb = [J_COLOR[j][0] / 255.0,
                        J_COLOR[j][1] / 255.0,
                        J_COLOR[j][2] / 255.0]

            ax.scatter(j_pos[0], j_pos[1], j_pos[2], s=3, color=c_rgb)
            ax.plot((j_pos[0], parent_j_pos[0]),
                    (j_pos[1], parent_j_pos[1]),
                    (j_pos[2], parent_j_pos[2]), linewidth=1, color=c_rgb)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(0, 4)

        plt.savefig(tmp_j_3d_dir + 'j3d' + idx + '.png')
        plt.close()

    return

def make_dpth_img(tmp_dir, dpth_mat):
    tmp_dir = tmp_dir + 'dpth/'
    os.makedirs(tmp_dir, exist_ok=True)

    for key in sorted(dpth_mat.keys())[:-3]:    # omit file mat explain
        img = dpth_mat[key]

        # normalize depth range without background
        max = img.max()
        min = img.min()
        img[max == img] = -1
        
        max = img.max()

        img = (img - min) / (max - min)
        img[img < 0] = 1.0
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        cv2.imwrite(tmp_dir + 'dpth' + key + '.jpg', img)

    return

def make_flow_img(tmp_dir, flow_mat):
    tmp_dir = tmp_dir + 'flow/'
    os.makedirs(tmp_dir, exist_ok=True)
    
    for key in sorted(flow_mat.keys())[:-3]:    # omit file mat explain
        flow = flow_mat[key]
        height, width = flow.shape[:2]

        # make image on HSV format, then change RGB
        hsv_img = np.zeros([height, width, 3], dtype=np.float32)

        for h in range(height):
            for w in range(width):
                norm = np.linalg.norm(flow[h, w], ord=2)
                if 0 != norm:
                    radian = np.arctan2(flow[h, w, 1], flow[h, w, 0])
                    hsv_img[h,w] = (radian, norm, 1.0)
                else:
                    hsv_img[h,w,1:] = (0.0,1.0)

        hsv_img[:,:,0] = np.rad2deg(hsv_img[:,:,0]) + 180
        satu_max = hsv_img[:,:,1].max()
        hsv_img[:,:,1] = hsv_img[:,:,1] / satu_max * 255
        hsv_img[:,:,2] = hsv_img[:,:,2] * 255
        hsv_img = hsv_img.astype(np.uint8)

        bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR_FULL)
        cv2.imwrite(tmp_dir + 'flow' + key + '.jpg', bgr_img)

    return

def make_nrml_img(tmp_dir, nrml_mat):
    tmp_dir = tmp_dir + 'nrml/'
    os.makedirs(tmp_dir, exist_ok=True)
    
    for key in sorted(nrml_mat.keys())[:-3]:    # omit file mat explain
        img = nrml_mat[key]

        img[np.all(0 == img, axis=2)] = 1.0
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        cv2.imwrite(tmp_dir + 'nrml' + key + '.jpg', img)

    return

SEGM_NUM = 25
SEGM_COLOR_LIST = [
    (255,255,255),(128,0,0),(255,0,0),(255,165,0),(238,232,170),(128,128,0),(255,255,0),(154,205,50),
    (85,107,47),(173,255,47),(0,255,255),(0,250,154),(32,178,170),(0,255,255),(72,209,204),(70,130,180),
    (30,144,255),(135,206,250),(0,0,128),(0,0,255),(138,43,226),(72,61,139),(148,0,211),(128,0,128),
    (221,160,221),(255,0,255),(255,20,147),(255,192,203),(255,250,205),(139,69,19),(210,105,30),(205,133,63),
    (188,143,143),(255,218,185),(245,255,250),(112,128,144),(176,196,222),(255,250,240),(240,255,240),
    (240,255,255),(128,128,128),(211,211,211),(220,20,60),(255,127,80),(255,69,0),(255,215,0),(189,183,107),
    (85,107,47),(127,255,0),(143,188,143),(102,205,170),(176,224,230),(106,90,205),(224,255,255),
]
def make_segm_img(tmp_dir, segm_mat):
    tmp_dir = tmp_dir + 'segm/'
    os.makedirs(tmp_dir, exist_ok=True)

    for key in sorted(segm_mat.keys())[:-3]:    # omit file mat explain
        segm = segm_mat[key]

        height, width = segm.shape[:2]
        img = np.zeros([height, width, 3], dtype=np.uint8)

        for segm_idx in range(SEGM_NUM):
            img[np.where(segm_idx == segm)] = SEGM_COLOR_LIST[segm_idx]

        cv2.imwrite(tmp_dir + 'segm' + key + '.jpg', img)

    return

def make_optional_img(tmp_dir):
    if os.path.isfile(tmp_dir + 'dpth.mat'):
        dpth_mat = scipy.io.loadmat(tmp_dir + 'dpth.mat')
        make_dpth_img(tmp_dir, dpth_mat)

    if os.path.isfile(tmp_dir + 'flow.mat'):
        flow_mat = scipy.io.loadmat(tmp_dir + 'flow.mat')
        make_flow_img(tmp_dir, flow_mat)

    if os.path.isfile(tmp_dir + 'nrml.mat'):
        nrml_mat = scipy.io.loadmat(tmp_dir + 'nrml.mat')
        make_nrml_img(tmp_dir, nrml_mat)

    if os.path.isfile(tmp_dir + 'segm.mat'):
        segm_mat = scipy.io.loadmat(tmp_dir + 'segm.mat')
        make_segm_img(tmp_dir, segm_mat)

    return

def main(dataset):
    output_dir = '../output/'

    tar = tarfile.open(dataset)
    tar.extractall(path=output_dir)
    tar.close()

    id = os.path.splitext(os.path.splitext(os.path.basename(dataset))[0])[0]
    tmp_dir = output_dir + id + '/'

    # make output image
    make_output_img(tmp_dir)

    # make optional image
    make_optional_img(tmp_dir)

    return


if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='Restore synthesic data.')
    parser.add_argument('--file', type=str, required=True, help='file path to dataset tar.gz')

    args = parser.parse_args()
    dataset = args.file

    main(dataset)