batch_size =4
patch_size = [1024, 1024]
epochs = 500
#6 0.001-0.0005
start_lr = 0.0001
lr_power = 0.9
weight_decay = 0.0001
num_worker = 0

alpha = 0.25
gamma = 2


backend = "retinanet"

# data_path = "../../train/"
train_sample_path = '../../data/t/'
val_img_path = "../../data/val/"
val_img_path_HPV = "../../data/valid_hpv/"
val_img_path_cluecell = "../../data/v/val_cluecell/"
val_img_path_3d = "../../data/v/3d_yangxing353/all/"
val_img_path_dst = "/mnt/data/0826/duxiaping/data/v/dst_0929/411403C09200705002-1/torch/"
val_gt_json_path = "../../data/val_gt/"
val_predict_json_path_allpast = '../json/val_allpast/'
val_predict_json_path_hpvpast = '../json/val_hpvpast/'
val_predict_json_path_allnow = '../json/val_allnow/'
val_predict_json_path_hpvnow = '../json/val_hpvnow/'
val_predict_json_path3 = '../json/val_cluecell/'
val_predict_json_path4 = '../json/val_3d/'
val_predict_json_path4_old = '../json/val_3d_old/'
val_predict_json_path5 = '../json/val_dst/'
val_predict_json_path5_old = '../json/val_dst_old/'


visual_sample_path = ""  # change to validation sample path (including .npz files)
# checkpoint_path = "../checkpoint_resize/"
checkpoint_path = "../checkpoint_mix_zhangxin/"
# checkpoint_path = "../c/"
log_path = "../log3/"
result_path = "../result/"

