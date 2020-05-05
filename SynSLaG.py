# coding: utf-8
import os, sys
import argparse
import time
import random
import bpy
import configparser
import math
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view as world2cam
from scipy.spatial.transform import Rotation
import scipy.io
import cv2
import numpy as np
import tarfile
import shutil
import collections
import json
import pickle

# constant
WORK_DIR = os.getcwd() + '/'
ACT_MATCH_DICT = {'Pelvis':'Hips',
    'L_Hip':'LeftHip', 'L_Knee':'LeftKnee', 'L_Ankle':'LeftAnkle', 'L_Foot':'LeftToe',
    'R_Hip':'RightHip', 'R_Knee':'RightKnee', 'R_Ankle':'RightAnkle', 'R_Foot':'RightToe',
    'Spine1':'Chest1', 'Spine2':'Chest2', 'Spine3':'Chest3',
    'Neck':'Neck', 'Head':'Head',
    'L_Collar':'LeftCollar', 'L_Shoulder':'LeftShoulder', 'L_Elbow':'LeftElbow', 'L_Wrist':'LeftWrist',
    'lindex0':'LeftIndex1', 'lindex1':'LeftIndex2', 'lindex2':'LeftIndex3',
    'lmiddle0':'LeftMiddle1', 'lmiddle1':'LeftMiddle2', 'lmiddle2':'LeftMiddle3',
    'lpinky0':'LeftPinky1', 'lpinky1':'LeftPinky2', 'lpinky2':'LeftPinky3',
    'lring0':'LeftRing1', 'lring1':'LeftRing2', 'lring2':'LeftRing3',
    'lthumb0':'LeftThumb1', 'lthumb1':'LeftThumb2', 'lthumb2':'LeftThumb3',
    'R_Collar':'RightCollar', 'R_Shoulder':'RightShoulder', 'R_Elbow':'RightElbow', 'R_Wrist':'RightWrist',
    'rindex0':'RightIndex1', 'rindex1':'RightIndex2', 'rindex2':'RightIndex3',
    'rmiddle0':'RightMiddle1', 'rmiddle1':'RightMiddle2', 'rmiddle2':'RightMiddle3',
    'rpinky0':'RightPinky1', 'rpinky1':'RightPinky2', 'rpinky2':'RightPinky3',
    'rring0':'RightRing1', 'rring1':'RightRing2', 'rring2':'RightRing3',
    'rthumb0':'RightThumb1', 'rthumb1':'RightThumb2', 'rthumb2':'RightThumb3'
}
BODY_PART_MATCH_DICT = {
    'bone_00':'Pelvis', 'bone_01':'L_Hip', 'bone_02':'R_Hip',
    'bone_03':'Spine1', 'bone_04':'L_Knee', 'bone_05':'R_Knee', 'bone_06':'Spine2',
    'bone_07':'L_Ankle', 'bone_08':'R_Ankle', 'bone_09':'Spine3', 'bone_10':'L_Foot', 'bone_11':'R_Foot',
    'bone_12':'Neck', 'bone_13':'L_Collar', 'bone_14':'R_Collar', 'bone_15':'Head',
    'bone_16':'L_Shoulder', 'bone_17':'R_Shoulder', 'bone_18':'L_Elbow', 'bone_19':'R_Elbow',
    'bone_20':'L_Wrist', 'bone_21':'R_Wrist', 'bone_22':'L_Hand', 'bone_23':'R_Hand'
}

SEGM_SORTED_PARTS = ['head', 'hips',
                        'leftArm', 'leftFoot', 'leftForeArm', 'leftHand', 'leftHandIndex1', 'leftLeg', 'leftShoulder', 'leftToeBase', 'leftUpLeg',
                        'neck',
                        'rightArm', 'rightFoot', 'rightForeArm', 'rightHand', 'rightHandIndex1', 'rightLeg', 'rightShoulder', 'rightToeBase', 'rightUpLeg',
                        'spine', 'spine1', 'spine2']
SEGM_PART2NUM = {part:(ipart+1) for ipart, part in enumerate(SEGM_SORTED_PARTS)}

# global variable
START_TIME = 0

def log_message(message):
    elapsed_time = time.time() - START_TIME
    print('[%.2f s] %s' %(elapsed_time, message))

def load_conf(file='./config', section='SYNTH_DATA'):
    """load configuration 
    
    Args:
        file (str, optional): path to conf file. Defaults to './config'.
        section (str, optional): name of section. Defaults to 'SYNTH_DATA'.
    
    Returns:
        [str]: params of configuration
    """
    log_message('Load configuration.')

    config = configparser.ConfigParser()
    resource = config.read(file)
    if 0 == resource:
        log_message('Error: cannot read configuration file.')
        exit(1)

    params = {}
    options = config.options(section)
    for opt in options:
        params[opt] = config.get(section, opt)
        log_message(' - %s: %s' % (opt, params[opt]))

    return params

def init_mat_tree(tree, spher_dir):
    """initial setting for material tree
    
    Args:
        tree (node_tree): node tree of material
        spher_dir (str): path to spherical harmonics
    
    Returns:
        [node_tree]: constructed tree
    """
    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    uv = tree.nodes.new('ShaderNodeTexCoord')
    uv.location = -800, 400

    uv_xform = tree.nodes.new('ShaderNodeVectorMath')
    uv_xform.location = -600, 400
    uv_xform.inputs[1].default_value = (0, 0, 1)
    uv_xform.operation = 'AVERAGE'

    uv_im = tree.nodes.new('ShaderNodeTexImage')
    uv_im.location = -400, 400

    rgb = tree.nodes.new('ShaderNodeRGB')
    rgb.location = -400, 200

    script = tree.nodes.new('ShaderNodeScript')
    script.location = -230, 400
    script.mode = 'EXTERNAL'
    script.filepath = spher_dir + 'sh.osl'
    script.update()

    # the emission node makes it independent of the scene lighting
    emission = tree.nodes.new('ShaderNodeEmission')
    emission.location = -60, 400

    mat_out = tree.nodes.new('ShaderNodeOutputMaterial')
    mat_out.location = 110, 400

    tree.links.new(uv.outputs[2], uv_im.inputs[0])
    tree.links.new(uv_im.outputs[0], script.inputs[0])
    tree.links.new(script.outputs[0], emission.inputs[0])
    tree.links.new(emission.outputs[0], mat_out.inputs[0])

    return tree

def init_scn_tree(tree, dpth_flg, segm_flg, nrml_flg, flow_flg):
    """Initial setting for scene tree
    
    Args:
        tree ([node_tree]): node tree of scene (background)
        dpth_flg (bool): optional output flags for depth
        segm_flg (bool): optional output for part segmentation
        nrml_flg (bool): optional output for surface normal
        flow_flg (bool): optional output for optical flow
    
    Returns:
        [node_tree]: constructed tree
    """
    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create node for foreground image
    layers = tree.nodes.new('CompositorNodeRLayers')
    layers.location = -300, 400

    # create node for background image
    bg_im = tree.nodes.new('CompositorNodeImage')
    bg_im.location = -300, 30

    # create node for mixing foreground and background images 
    mix = tree.nodes.new('CompositorNodeMixRGB')
    mix.location = 40, 30
    mix.use_alpha = True

    # create node for the final output (RGB image)
    composite_out = tree.nodes.new('CompositorNodeComposite')
    composite_out.location = 240, 30

    # create node for depth output
    if(dpth_flg):
        dpth_out = tree.nodes.new('CompositorNodeOutputFile')
        dpth_out.location = 240, 700
        dpth_out.format.file_format = 'OPEN_EXR'
        dpth_out.name = 'Depth Output'
    
    # create node for normal output
    if(nrml_flg):
        nrml_out = tree.nodes.new('CompositorNodeOutputFile')
        nrml_out.location = 240, 600
        nrml_out.format.file_format = 'OPEN_EXR'
        nrml_out.name = 'Normal Output'
    
    # create node for optical flow
    if(flow_flg):
        flow_out = tree.nodes.new('CompositorNodeOutputFile')
        flow_out.location = 240, 500
        flow_out.format.file_format = 'OPEN_EXR'
        flow_out.name = 'Flow Output'
    
    # create node for part segmentation
    if(segm_flg):
        segm_out = tree.nodes.new('CompositorNodeOutputFile')
        segm_out.location = 240, 400
        segm_out.format.file_format = 'OPEN_EXR'
        segm_out.name = 'Segmentation Output'
        
    # merge fg and bg images
    tree.links.new(bg_im.outputs[0], mix.inputs[1])
    tree.links.new(layers.outputs['Image'], mix.inputs[2])
    tree.links.new(mix.outputs[0], composite_out.inputs[0])

    if(dpth_flg):
        tree.links.new(layers.outputs['Depth'], dpth_out.inputs[0])
    if(nrml_flg):
        tree.links.new(layers.outputs['Normal'], nrml_out.inputs[0])
    if(flow_flg):
        tree.links.new(layers.outputs['Vector'], flow_out.inputs[0])
    if(segm_flg):
        tree.links.new(layers.outputs['IndexMA'], segm_out.inputs[0])

    return tree

def init_scene(gender, model_dir, spher_dir, x_res, y_res,
                dpth_flg, segm_flg, nrml_flg, flow_flg):
    """Initialize scene.
    
    Args:
        gender (str): gender -> 'female' or 'male'
        model_dir (str): path to base model directory
        spher_dir (str): path to spherical harmonics directory
        x_res (int): rendering resolution for width
        y_res (int): rendering resolution for height
        dpth_flg (bool): optional output flags for depth
        segm_flg (bool): optional output for part segmentation
        nrml_flg (bool): optional output for surface normal
        flow_flg (bool): optional output for optical flow
    
    Returns:
        [obj]: Scene object
        [obj]: Scene tree
        [obj]: Material tree
        [str]: Body object name
        [obj]: Armature object
        [obj]: Camera object 1
        [dict]: Materials for parts segmentation
    """
    log_message('Initialize scene.')

    # load fbx model
    bpy.ops.import_scene.fbx(filepath=model_dir + ('%s_jsl_model_v123.fbx' % gender[0]),
                             axis_forward='-Z', axis_up='Y', global_scale=100)

    # delete the default cube (which held the material)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete(use_global=False)

    # set Rendering scene
    scene = bpy.data.scenes['Scene']
    scene.render.engine = 'CYCLES'
    scene.cycles.shading_system = True

    scene.cycles.film_transparent = True
    scene.render.layers["RenderLayer"].use_pass_vector = True
    scene.render.layers["RenderLayer"].use_pass_normal = True
    scene.render.layers['RenderLayer'].use_pass_emit  = True
    scene.render.layers['RenderLayer'].use_pass_emit  = True
    scene.render.layers['RenderLayer'].use_pass_material_index  = True

    # set Scene tree
    scene.use_nodes = True
    scn_tree = init_scn_tree(scene.node_tree, dpth_flg, segm_flg, nrml_flg, flow_flg)

    # set Material tree
    material = bpy.data.materials['Material']
    material.use_nodes = True
    mat_tree = init_mat_tree(material.node_tree, spher_dir)

    # set body object param
    body_name = '%s_avg' % gender[0]
    body_ob = bpy.data.objects[body_name]
    body_ob.data.use_auto_smooth = False
    body_ob.active_material = material
    body_ob.data.shape_keys.animation_data_clear()

    # unblocking both the pose and the blendshape limits
    for k in body_ob.data.shape_keys.key_blocks.keys():
        bpy.data.shape_keys["Key"].key_blocks[k].slider_min = -10
        bpy.data.shape_keys["Key"].key_blocks[k].slider_max = 10

    if segm_flg:
        # create parts segmentation
        segm_materials = create_segmentation(scene, body_ob, model_dir)
    else:
        segm_materials = None

    # clear existing animation data
    arm_ob = bpy.data.objects['Armature']
    arm_ob.animation_data_clear()

    # set camera
    bpy.ops.object.select_all(action='DESELECT')
    cam_ob1 = bpy.data.objects['Camera']
    scn = bpy.context.scene
    scn.objects.active = cam_ob1

    cam_ob1.matrix_world = Matrix(((1, 0., 0., 0.),
                                   (0., 0., -1, -5),
                                   (0., 1., 0., 1),
                                   (0.0, 0.0, 0.0, 1.0)))
    cam_ob1.data.angle = math.radians(40)
    cam_ob1.data.lens =  60
    cam_ob1.data.clip_start = 0.1
    cam_ob1.data.clip_end = 350
    cam_ob1.data.sensor_width = 32

    # set render size
    scn.render.resolution_x = x_res
    scn.render.resolution_y = y_res
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = 'PNG'

    return scene, scn_tree, mat_tree, body_name, arm_ob, cam_ob1, segm_materials

def get_tmp_prop_list(gender, location, bg_dir, action_dir, texture_dir, shape_dir, body_name):
    """Get list of temporary property 
    
    Args:
        gender (str): gender -> 'male' or 'female'
        location (str): location. 'inside' or 'outside'.
        bg_dir (str): path to background image
        action_dir (str): path to action file
        texture_dir (str): path to texture image
        shape_dir (str): path to body shape file
        body_name (str): name of body
    
    Returns:
        str: path list to background image
        str: path list to action file
        str: path list to texture image
        array: array list of body shape
    """
    log_message('Get temporary property list.')

    # load background files path
    bg_file = bg_dir + ('%s.txt' % location)
    with open(bg_file, 'r') as f:
        bg_list = f.read().splitlines()
    random.shuffle(bg_list)

    # load action files path
    action_file = action_dir + ('%s.txt' % gender)
    with open(action_file, 'r') as f:
        action_list = f.read().splitlines()

    # load texture files
    texture_file = texture_dir + ('%s_all.txt' % gender)
    with open(texture_file, 'r') as f:
        texture_list = f.read().splitlines()
    random.shuffle(texture_list)

    # load body shape array
    # compute the number of shape blendshapes in the model
    body_ob = bpy.data.objects[body_name]
    n_sh_bshapes = len([k for k in body_ob.data.shape_keys.key_blocks.keys()
                        if k.startswith('Shape')])
    smpl_data = np.load(shape_dir + 'smpl_data.npz')
    body_shape_2darray = smpl_data['%sshapes' % gender][:, :n_sh_bshapes]
    body_shape_list = []
    for shape_array in body_shape_2darray:
        body_shape_list.append(shape_array)
    random.shuffle(body_shape_list)
    # body_shape_list = []
    # body_shape_list.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))    # average
    # body_shape_list.append(np.array([ 2.25176191, -3.7883464 ,  0.46747496,  3.89178988,  2.20098416,  0.26102114, -3.07428093,  0.55708514, -3.94442258, -2.88552087]))    # fat
    # body_shape_list.append(np.array([-2.26781107,  0.88158132, -0.93788176, -0.23480508,  1.17088298,  1.55550789,  0.44383225,  0.37688275, -0.27983086,  1.77102953]))    # thin
    # body_shape_list.append(np.array([ 0.00404852,  0.8084637 ,  0.32332591, -1.33163664,  1.05008727,  1.60955275,  0.22372946, -0.10738459,  0.89456312, -1.22231216]))    # short
    # body_shape_list.append(np.array([ 3.63453289,  1.20836171,  3.15674431, -0.78646793, -1.93847355, -0.32129994, -0.97771656,  0.94531640,  0.52825811, -0.99324327]))    # tall

    return bg_list, action_list, texture_list, body_shape_list

def get_joints_pos(scene, body_name, arm_ob, cam_ob1):
    """Get 3D and 2D joints position.
    
    Args:
        scene (object): scene
        body_name (str): name of body object
        arm_ob (object): armature
        cam_ob1 (object): camera
    
    Returns:
        dict: dictionary of 3D joints position
        dict: dictionary of 2D joints position
    """
    # get normalization measure for 3D position
    l_sh_pos = arm_ob.matrix_world * arm_ob.pose.bones[('%s_L_Shoulder' % body_name)].head
    r_sh_pos = arm_ob.matrix_world * arm_ob.pose.bones[('%s_R_Shoulder' % body_name)].head
    norm_measure = 1 / (np.linalg.norm(l_sh_pos - r_sh_pos, ord=2))

    # set 3D and 2D joints position
    joints3D = collections.OrderedDict()
    joints2D = collections.OrderedDict()

    for bone in arm_ob.pose.bones:
        if 'root' in bone.name:
            continue

        bone_name = bone.name.lstrip(body_name + '_')

        b_3d_pos = arm_ob.matrix_world * bone.head
        joints3D[bone_name] = (b_3d_pos.x * norm_measure,   # normalize position
                                b_3d_pos.y * norm_measure,
                                b_3d_pos.z * norm_measure)

        b_2d_pos = world2cam(scene, cam_ob1, arm_ob.matrix_world * bone.head)
        joints2D[bone_name] = (b_2d_pos.x, 1 - b_2d_pos.y)  # set root to upper-left

    return joints3D, joints2D

def modify_blend_shapes(arm_ob, body_name, body_shape, frame=None):
    """modify blend shapes for pose and body,
        then push the bshapes to frame if it gets frame.
    
    Args:
        arm_ob (obj): armature object
        body_name (str): body name
        body_shape (array):  body shape array
        frame (frame, optional): frame of animation
    """
    body_ob = bpy.data.objects[body_name]

    # get pose shape for body bone positions
    pose_shape_list = []
    for ibone in range(len(BODY_PART_MATCH_DICT)):
        bone = arm_ob.pose.bones[body_name + '_' + BODY_PART_MATCH_DICT['bone_%02d' % ibone]]
        quaternion = bone.rotation_quaternion
        rot_mat = Rotation.from_quat(np.array([quaternion.x,
                                                quaternion.y,
                                                quaternion.z,
                                                quaternion.w])).as_matrix()
        pose_shape_list.append((rot_mat - np.eye(3)).ravel())

    pose_shape = np.concatenate([p_sh for p_sh in pose_shape_list[1:]])    # except Pelvis(bone_00) shapes

    # apply pose blend shapes
    for i_pose_sh, pose_sh_elem in enumerate(pose_shape):
        body_ob.data.shape_keys.key_blocks['Pose%03d' % i_pose_sh].value = pose_sh_elem
        if frame is not None:
            body_ob.data.shape_keys.key_blocks['Pose%03d' % i_pose_sh].keyframe_insert('value', index=-1, frame=frame)

    # apply body shape blendshapes
    for i_body_sh, body_sh_elem in enumerate(body_shape):
        body_ob.data.shape_keys.key_blocks['Shape%03d' % i_body_sh].value = body_sh_elem
        if frame is not None:
            body_ob.data.shape_keys.key_blocks['Shape%03d' % i_body_sh].keyframe_insert('value', index=-1, frame=frame)

    return

def select_children_object(obj_name):
    """select onjects include children
    
    Args:
        obj_name (str): target object name
    """
    obj = bpy.data.objects[obj_name]
    obj.select = True

    for child_obj in obj.children:
        select_children_object(child_obj.name)
    
    return

def set_action2arm(scene, action, arm_ob, body_name, body_shape, gender, offset_rot_mat_dict):
    """ set fbx action to armature
    
    Args:
        scene (obj): scene object
        action (str): path to action fbx file
        arm_ob (obj): armature object
        body_name (str): name of body mesh
        body_shape (array): body shape array
        gender (str): gender -> female or male
        offset_rot_mat_dict (dictionary): rotation matrix to offset pose from reset pose
    """
    log_message('Set action to body model.')

    bpy.ops.import_scene.fbx(filepath=action,
        axis_forward='-Z', axis_up='Y', global_scale=1.0,
        use_custom_normals=False, use_image_search=False,
        use_anim=True, use_custom_props=False)

    act_ob = bpy.data.objects['Root']
    f_start, f_end = act_ob.animation_data.action.frame_range
    scene.frame_start = f_start
    scene.frame_end = f_end

    for frame in range(scene.frame_end):
        scene.frame_set(frame)
        arm_ob.pose.bones[body_name + '_root'].keyframe_insert('location', frame=frame)

        # set offset pose before load action
        for key, value in offset_rot_mat_dict.items():
            bone = arm_ob.pose.bones[key]
            quaternion = Matrix(value).to_quaternion()
            bone.rotation_quaternion = quaternion

        for key, value in ACT_MATCH_DICT.items():
            bone = arm_ob.pose.bones[body_name + '_' + key]
            quaternion = act_ob.pose.bones[value].rotation_quaternion * bone.rotation_quaternion
            bone.rotation_quaternion = quaternion
            bone.keyframe_insert('rotation_quaternion', frame=frame)
            bone.keyframe_insert('location', frame=frame)

        # modify blend shapes for pose
        modify_blend_shapes(arm_ob, body_name, body_shape, frame)

    # remove action data
    # automatically be selected action objects when there was loaded from fbx file
    bpy.ops.object.delete(use_global=False)

    return

def set_shape2arm(scene, body_name, arm_ob,
                    body_shape, body_reg_ivs, body_joint_reg,
                    l_hand_reg_verts_dict, r_hand_reg_verts_dict):
    """set various body shape and modify bones to shape
    
    Args:
        scene (obj): scene object
        body_name (str): name of body and mesh object
        arm_ob (obj): armature object
        body_shape (array): body shape array
        body_reg_ivs (array): vertices index for body regression
        body_joint_reg (array): joints array for body regression
        l_hand_reg_verts_dict (dict): vertices array's dictionary for left hand regression
        r_hand_reg_verts_dict (dict): vertices array's dictionary for right hand regression
    """
    log_message('Set body shape and fitting bones.')

    scene.objects.active = arm_ob
    body_ob = bpy.data.objects[body_name]
    body_reg_vs = np.empty((len(body_reg_ivs), 3))

    # set arm bones to reset position
    reset_mat, _ = cv2.Rodrigues(np.array([0.0, 0.0, 0.0]))
    reset_quaternion = Matrix(reset_mat).to_quaternion()
    for arm_bone in arm_ob.pose.bones:
        arm_bone.rotation_quaternion = reset_quaternion

    # modify blend shapes for pose
    modify_blend_shapes(arm_ob, body_name, body_shape)

    # obtain a mesh after applying modifiers
    bpy.ops.wm.memory_statistics()
    # me holds the vertices after applying the shape blendshapes
    mesh = body_ob.to_mesh(scene, True, 'PREVIEW')

    # fill the body regressor vertices matrix
    for iiv, iv in enumerate(body_reg_ivs):
        body_reg_vs[iiv] = mesh.vertices[iv].co
    # regress body joint positions in rest pose
    body_joint_xyz = body_joint_reg.dot(body_reg_vs)

    # get left hand regression joints position
    l_hand_joint_xyz_dict = {}
    for joint, v_indexes in l_hand_reg_verts_dict.items():
        sum_joint_vec = Vector([0.0, 0.0, 0.0])
        for v_index in v_indexes:
            sum_joint_vec = sum_joint_vec + mesh.vertices[v_index].co
        l_hand_joint_xyz_dict[joint] = sum_joint_vec / v_indexes.shape[0]

    # get right hand regression joints position
    r_hand_joint_xyz_dict = {}
    for joint, v_indexes in r_hand_reg_verts_dict.items():
        sum_joint_vec = Vector([0.0, 0.0, 0.0])
        for v_index in v_indexes:
            sum_joint_vec = sum_joint_vec + mesh.vertices[v_index].co
        r_hand_joint_xyz_dict[joint] = sum_joint_vec / v_indexes.shape[0]

    bpy.data.meshes.remove(mesh)

    # adapt body joint positions in rest pose
    bpy.ops.object.mode_set(mode='EDIT')
    for ibone in range(len(BODY_PART_MATCH_DICT)):
        bb = arm_ob.data.edit_bones[body_name + '_' + BODY_PART_MATCH_DICT['bone_%02d' % ibone]]
        bb_offset = bb.tail - bb.head
        bb.head = body_joint_xyz[ibone]
        bb.tail = bb.head + bb_offset
    # regress hands joint positions in rest pose
    # adapt hands joint positions in rest pose
    for key, value in l_hand_joint_xyz_dict.items():
        bb = arm_ob.data.edit_bones[key]
        bb_offset = bb.tail - bb.head
        bb.head = value
        bb.tail = bb.head + bb_offset
    for key, value in r_hand_joint_xyz_dict.items():
        bb = arm_ob.data.edit_bones[key]
        bb_offset = bb.tail - bb.head
        bb.head = value
        bb.tail = bb.head + bb_offset
    bpy.ops.object.mode_set(mode='OBJECT')

    return

def create_segmentation(scene, body_ob, model_dir):
    # create one material per part as defined in a pickle with the segmentation
    # this is useful to render the segmentation in a material pass
    scene.objects.active = body_ob

    materials = {}
    vgroups = {}

    # with open(model_dir + 'segm_per_v_overlap_SMPLH.pkl', 'rb') as f:
    with open(model_dir + 'segm_per_v_overlap.pkl', 'rb') as f:
        vsegm = pickle.load(f)
    parts = sorted(vsegm.keys())

    for part in parts:
        vs = vsegm[part]
        
        vgroups[part] = body_ob.vertex_groups.new(part)
        vgroups[part].add(vs, 1.0, 'ADD')
        bpy.ops.object.vertex_group_set_active(group=part)
        
        materials[part] = bpy.data.materials['Material'].copy()
        materials[part].pass_index = SEGM_PART2NUM[part]
        bpy.ops.object.material_slot_add()
        body_ob.material_slots[-1].material = materials[part]

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.vertex_group_select()
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode='OBJECT')

        materials[part].node_tree.nodes['Script'].update()

    return materials

def generate_dataset(scene, mat_tree, scn_tree, body_name, arm_ob, cam_ob1,
        gender, texture, bg, action,
        body_shape, body_reg_ivs, body_joint_reg,
        l_hand_reg_verts_dict, r_hand_reg_verts_dict, offset_rot_mat_dict,
        h_img, w_img, fps, tmp_dir, output_dir,
        dpth_flg, segm_flg, nrml_flg, flow_flg, segm_materials):
    """Generate dataset.
    
    Args:
        scene (object): Scene
        mat_tree (node_tree): node tree of material
        scn_tree (node_tree): node tree of scene
        body_name (str): name of body object
        arm_ob (object): armature object
        cam_ob1 (object): camera object 1
        gender (str): gender-> female or male
        texture (str): path to texture image
        bg (str): path to background image
        action (str): path to action file(bvh)
        body_shape(array): body shape array
        body_reg_ivs(array): body vertices for regression
        body_joint_reg(array): body joint position for regression
        l_hand_reg_verts_dict(dictionary): left hand regressor dictionary
        r_hand_reg_verts_dict(dictionary): right hand regressor dictionary
        offset_rot_mat_dict(dictionary): rotation matrix to offset pose from reset pose
        h_img (int): height of background image
        w_img (int): width of background image
        fps (int): frame per second for rendering
        tmp_dir (str): path to tmporary directory
        output_dir (str): path to output directory
        dpth_flg (bool): optional output flags for depth
        segm_flg (bool): optional output for part segmentation
        nrml_flg (bool): optional output for surface normal
        flow_flg (bool): optional output for optical flow
        segm_materials (dict): Materials for parts segmentation
    """
    # make directories for processing
    output_dir = WORK_DIR + output_dir
    os.makedirs(output_dir, exist_ok=True)
    id = ('%s_%s_%s' % (os.path.splitext(os.path.basename(action))[0],
                        os.path.splitext(os.path.basename(texture))[0],
                        os.path.splitext(os.path.basename(bg))[0]))
    tmp_dir = WORK_DIR + tmp_dir + id + '/'
    os.makedirs(tmp_dir, exist_ok=True)

    tmp_img_dir = tmp_dir + 'image/'
    os.makedirs(tmp_img_dir, exist_ok=True)

    # set optional output dir
    tmp_dpth_dir = tmp_dir + 'depth/'
    tmp_nrml_dir = tmp_dir + 'normal/'
    tmp_flow_dir = tmp_dir + 'flow/'
    tmp_segm_dir = tmp_dir + 'segment/'
    if(dpth_flg):
        scn_tree.nodes['Depth Output'].base_path = tmp_dpth_dir
    if(nrml_flg):
        scn_tree.nodes['Normal Output'].base_path = tmp_nrml_dir
    if(flow_flg):
        scn_tree.nodes['Flow Output'].base_path = tmp_flow_dir
    if(segm_flg):
        scn_tree.nodes['Segmentation Output'].base_path = tmp_segm_dir

    # Set temprorary property
    log_message('Set temprary property.')

    texture_img = bpy.data.images.load(WORK_DIR + texture)
    mat_tree.nodes['Image Texture'].image = texture_img

    if None != segm_materials:
        for material in segm_materials.values():
            material.node_tree.nodes['Image Texture'].image = texture_img

    bg_img = bpy.data.images.load(WORK_DIR + bg)
    scn_tree.nodes['Image'].image = bg_img

    set_shape2arm(scene, body_name, arm_ob,
                    body_shape, body_reg_ivs, body_joint_reg,
                    l_hand_reg_verts_dict, r_hand_reg_verts_dict)

    set_action2arm(scene, action, arm_ob, body_name, body_shape, gender, offset_rot_mat_dict)

    # random light
    sh_coeffs = .7 * (2 * np.random.rand(9) - 1)
    sh_coeffs[0] = .5 + .9 * np.random.rand() # Ambient light (first coeff) needs a minimum  is ambient. Rest is uniformly distributed, higher means brighter.
    sh_coeffs[1] = -.7 * np.random.rand()
    
    for ish, coeff in enumerate(sh_coeffs):
        mat_tree.nodes['Script'].inputs[ish+1].default_value = coeff
    mat_tree.nodes['Vector Math'].inputs[1].default_value[:2] = (0,0) # reset material vector
    mat_tree.nodes['Script'].update()

    if None != segm_materials:
        for material in segm_materials.values():
            for ish, coeff in enumerate(sh_coeffs):
                material.node_tree.nodes['Script'].inputs[ish+1].default_value = coeff
            material.node_tree.nodes['Vector Math'].inputs[1].default_value[:2] = (0,0) # reset material vector
            material.node_tree.nodes['Script'].update()

    log_message('Start to generate dataset.')
    # get rendering properties
    frame_num = scene.frame_end
    scn_org_fps = scene.render.fps
    step_size = math.ceil(scn_org_fps / fps)

    log_message(' - rendering id: %s' % id)
    log_message(' - action: %s' % action)
    log_message(' - texture: %s' % texture)
    log_message(' - background: %s' % bg)
    log_message(' - body_shape: %s' % body_shape)
    log_message(' - sh_coeffs: %s' % sh_coeffs)
    log_message(' - fps: %s' % fps)

    with open(tmp_dir + 'prop_info', 'w') as f:
        f.write('[PROP_INFO]\n')
        f.write('action = %s\n' % action)
        f.write('texture = %s\n' % texture)
        f.write('background = %s\n' % bg)
        f.write('body_shape = %s\n' % body_shape)
        f.write('sh_coeff = %s\n' % sh_coeffs)
        f.write('fps = %s\n' % fps)

    # do rendering
    log_message('Start rendering.')

    render_num = int(frame_num / step_size)
    keypoints_dict = []

    dict_dpth = {}
    dict_nrml = {}
    dict_flow = {}
    dict_segm = {}

    for index in range(render_num):
        frame = index * step_size + 1
        scene.frame_set(frame)
        scene.render.use_antialiasing = False

        # disable render output
        logfile = tmp_dir + 'null'
        open(logfile, 'w').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)
        
        # Render
        scene.render.filepath = tmp_img_dir + ('img%08d.png' % index)
        bpy.ops.render.render(write_still=True)
        
        # disable output redirection
        os.close(1)
        os.dup(old)
        os.close(old)
        os.remove(logfile)

        # make 2D and 3D keypoints dataset dictionary
        joints3D, joints2D = get_joints_pos(scene, body_name, arm_ob, cam_ob1)
        keypoints = collections.OrderedDict()
        keypoints = {'image': 'image/' + ('img%08d.png' % index), 'joints3D': joints3D, 'joints2D': joints2D}
        keypoints_dict.append(keypoints)

        # extract optional output to matrix
        if(dpth_flg):
            render_img = bpy.data.images.load(tmp_dpth_dir + ('Image%04d.exr' % frame))
            # render_img.pixels size is width * height * 4 (rgba)
            arr = np.array(render_img.pixels[:]).reshape(h_img, w_img, 4)[::-1] # images are vertically flipped
            mat_dpth = arr[:, :, 0]
            dict_dpth['%08d' % index] = mat_dpth.astype(np.float32, copy=False)
            render_img.user_clear()
            bpy.data.images.remove(render_img)
        if(nrml_flg):
            render_img = bpy.data.images.load(tmp_nrml_dir + ('Image%04d.exr' % frame))
            # render_img.pixels size is width * height * 4 (rgba)
            arr = np.array(render_img.pixels[:]).reshape(h_img, w_img, 4)[::-1] # images are vertically flipped
            mat_nrml = arr[:, :, :3]
            dict_nrml['%08d' % index] = mat_nrml.astype(np.float32, copy=False)
            render_img.user_clear()
            bpy.data.images.remove(render_img)
        if(flow_flg):
            render_img = bpy.data.images.load(tmp_flow_dir + ('Image%04d.exr' % frame))
            # render_img.pixels size is width * height * 4 (rgba)
            arr = np.array(render_img.pixels[:]).reshape(h_img, w_img, 4)[::-1] # images are vertically flipped
            mat_flow = arr[:, :, 1:3]
            dict_flow['%08d' % index] = mat_flow.astype(np.float32, copy=False)
            render_img.user_clear()
            bpy.data.images.remove(render_img)
        if(segm_flg):
            render_img = bpy.data.images.load(tmp_segm_dir + ('Image%04d.exr' % frame))
            # render_img.pixels size is width * height * 4 (rgba)
            arr = np.array(render_img.pixels[:]).reshape(h_img, w_img, 4)[::-1] # images are vertically flipped
            mat_segm = arr[:, :, 0]
            dict_segm['%08d' % index] = mat_segm.astype(np.uint8, copy=False)
            render_img.user_clear()
            bpy.data.images.remove(render_img)

        log_message(' - rendered of %s/%s.' % (index+1, render_num))

    # save video and dataset
    log_message('Save dataset %s.' % (output_dir + id))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(output_dir + id + '.mp4', fourcc, fps, (w_img, h_img))
    for i in range(render_num):
        img = cv2.imread(tmp_img_dir + ('img%08d.png' % i))
        video.write(img)
    video.release()

    # save keypints dataset
    with open(tmp_dir + 'keypoints.json', 'w') as f:
        json.dump(keypoints_dict, f, indent=4)

    # save optional output files
    if(dpth_flg):
        scipy.io.savemat(tmp_dir + 'dpth.mat', dict_dpth, do_compression=True)
        shutil.rmtree(tmp_dpth_dir)
    if(nrml_flg):
        scipy.io.savemat(tmp_dir + 'nrml.mat', dict_nrml, do_compression=True)
        shutil.rmtree(tmp_nrml_dir)
    if(flow_flg):
        scipy.io.savemat(tmp_dir + 'flow.mat', dict_flow, do_compression=True)
        shutil.rmtree(tmp_flow_dir)
    if(segm_flg):
        scipy.io.savemat(tmp_dir + 'segm.mat', dict_segm, do_compression=True)
        shutil.rmtree(tmp_segm_dir)

    tar_name = output_dir + id + '.tar.gz'
    archive = tarfile.open(tar_name, mode='w:gz')
    archive.add(tmp_dir, arcname=id)
    archive.close()

    shutil.rmtree(tmp_dir)

    # reset temporary property
    arm_ob.animation_data_clear()
    bpy.data.images.remove(bg_img)
    bpy.data.images.remove(texture_img)

    return

def main(gender, location):
    """ generate synthesic data
    
    Args:
        gender ([str]): gender
        location ([str]): location
    """
    log_message('Start generation process.')

    # load configuration
    conf_params = load_conf('./config', 'SYNTH_DATA')

    bg_dir = conf_params['bg_dir']
    action_dir = conf_params['action_dir']
    texture_dir = conf_params['texture_dir']
    model_dir = conf_params['model_dir']
    shape_dir = conf_params['shape_dir']
    spher_dir = conf_params['spher_dir']

    tmp_dir = conf_params['tmp_dir']
    output_dir = conf_params['output_dir']
    
    iterate = int(conf_params['iterate'])
    h_img = int(conf_params['h_img'])
    w_img = int(conf_params['w_img'])    
    fps = int(conf_params['fps'])

    dpth_flg = bool(int(conf_params['depth']))
    segm_flg = bool(int(conf_params['part_segmentation']))
    nrml_flg = bool(int(conf_params['surface_normal']))
    flow_flg = bool(int(conf_params['optical_flow']))

    x_res = w_img
    y_res = h_img

    # initialize scene
    scene, scn_tree, mat_tree, body_name, arm_ob, cam_ob1, segm_materials = init_scene(
            gender, model_dir, spher_dir, x_res, y_res,
            dpth_flg, segm_flg, nrml_flg, flow_flg)

    # get temporary property list
    bg_list, action_list, texture_list, body_shape_list = get_tmp_prop_list(
            gender, location, bg_dir, action_dir, texture_dir, shape_dir, body_name)

    # load body shape regressor
    smpl_data = np.load(shape_dir + 'smpl_data.npz')
    body_reg_ivs = smpl_data['regression_verts']
    body_joint_reg = smpl_data['joint_regressor']

    # load hands shape regressor
    l_hand_reg_data = np.load(shape_dir + body_name + '_L_Hand_verts.npz')
    l_hand_reg_verts_dict = {}
    for index in l_hand_reg_data.files:
        l_hand_reg_verts_dict[index] = l_hand_reg_data[index]

    r_hand_reg_data = np.load(shape_dir + body_name + '_R_Hand_verts.npz')
    r_hand_reg_verts_dict = {}
    for index in r_hand_reg_data.files:
        r_hand_reg_verts_dict[index] = r_hand_reg_data[index]

    # load offset pose rotation matrix for fitting jsl dataset
    offset_rot_mat = np.load(shape_dir + body_name + '_Offset_Pose_Rotation.npz')
    offset_rot_mat_dict = {}
    for index in offset_rot_mat.files:
        offset_rot_mat_dict[index] = offset_rot_mat[index]
    
    # generate temporary dataset
    for i in range(iterate):
        it = i % len(texture_list)
        ib = i % len(bg_list)
        ia = i % len(action_list)
        ibs = i % len(body_shape_list)

        generate_dataset(scene, mat_tree, scn_tree, body_name, arm_ob, cam_ob1,
                gender, texture_list[it], bg_list[ib], action_list[ia],
                body_shape_list[ibs], body_reg_ivs, body_joint_reg,
                l_hand_reg_verts_dict, r_hand_reg_verts_dict, offset_rot_mat_dict,
                h_img, w_img, fps, tmp_dir, output_dir,
                dpth_flg, segm_flg, nrml_flg, flow_flg, segm_materials)

    return

if '__main__' == __name__:
    # time for logging
    START_TIME = time.time()
    log_message('Start main process.')

    # parse commandline arguments
    parser = argparse.ArgumentParser(description='Generate synthesic data.')
    parser.add_argument('--gender', type=str, choices=['female', 'male'], required=True,
        help='gender: female or male')
    parser.add_argument('--location', type=str, choices=['inside', 'outside'], required=True,
        help='location: inside or outside')
    
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    gender = args.gender
    location = args.location

    log_message(' - args gender: %s' % gender)
    log_message(' - args location: %s' % location)

    # call main function
    main(gender, location)
