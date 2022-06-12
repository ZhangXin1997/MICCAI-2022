import json
from glob import glob
import time
import os
import numpy as np
import cfg
from util import *

def prepare_gt_data(path):
    """
    将label中的标注转化为以roi为单位的数据,生成的每个json文件为某个roi的ground truth bbox, 而这里每个json文件的命名由2部分组成
    1,这个roi属于哪个kfb  2,这个roi是这个kfb的第几个roi
    :param path: 要转化的数据所在的路径
    :return:
    """
    filenames=os.listdir(os.path.join(path,"labels"))
    for filename in filenames:
        json_path = glob(os.path.join(path, "labels", filename))[0]
        with open(json_path, 'r') as f:
            json_infos = json.loads(f.read())

        """下面的作用是根据label中的json文件,找到这个json文件中的所有roi的坐标,并且将这些坐标保存在roi_coords数组中"""
        roi_coords = []
        for json_info in json_infos:
            if json_info['class'] == 'roi':
                coord = {'x': json_info['x'], 'y': json_info['y'], 'w': json_info['w'], 'h': json_info['h']}
                roi_coords.append(coord)

        """下面的作用是对每个roi,计算它包含的bbox的坐标,每个roi的bbox保存一个json文件中,即每个roi都有这样的一个json文件"""
        roi_cnt = 1 #roi_cnt表示当前的roi是它所属的kfb的第几个roi
        """下面的for循环,以循环的方式为每个roi找到它对应的bbox"""
        for roi_coord in roi_coords:
            cur_roi_name=filename.split(".")[0]+"_"+str(roi_cnt)+".json" #cur_roi_name是当前这个roi保存为json的文件的名字,即名字中包含1,这个roi属于哪个kfb文件 2, 这个roi是这个kfb的第几个roi
            X, Y, W, H = roi_coord['x'], roi_coord['y'], roi_coord['w'], roi_coord['h']
            bboxs=[] #用来保存当前roi的bbox
            pos_cnt = 0
            for json_info in json_infos:
                """判断哪些bbox属于当前的roi"""
                if json_info['class'] == 'pos':
                    x, y, w, h = json_info['x'], json_info['y'], json_info['w'], json_info['h']
                    if X < x < X + W and Y < y < Y + H:
                        pos_cnt += 1
                        box={} #box中含有的关键字是xywh,而不是xyxy
                        box["x"] = max(int(x - X), 0)
                        box["y"] = max(int(y - Y), 0)
                        box["w"] = min(int(W - 1 + X - x), w)
                        box["h"] = min(int(H - 1 + Y - y), h)
                        bboxs.append(box)

            if pos_cnt == 0:
                continue

            eval_gt_data_path = cfg.eval_gt_data_path
            mkdir(eval_gt_data_path)
            final_path=os.path.join(eval_gt_data_path,cur_roi_name)
            """下面是将当前的roi的所有bbox保存为json文件"""
            with open(final_path,"w") as f:
                json.dump(bboxs,f)

            roi_cnt += 1
        print("Finish: ", filename, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

def xywh2xyxy(array):
    newarray=np.empty(array.shape)
    newarray[:, :2] = array[:, :2]
    newarray[:, 2:] = array[:, 2:] + array[:, :2]
    return newarray

def compute_iou(fir_array, sec_array):
    """
    这里的作用是计算iou, 最终得到一个mxn的数组, m是fir_array中bbox的数量, n是sec_array中bbox的数量
    :param fir_array: shape为mx4, style为xyxy
    :param sec_array: shape为nx4, style为xyxy
    :return: 一个m*n的数组
    """
    fir_area = (fir_array[:, 2] - fir_array[:, 0]) * (fir_array[:, 3] - fir_array[:, 1])
    sec_area = (sec_array[:, 2] - sec_array[:, 0]) * (sec_array[:, 3] - sec_array[:, 1])

    """maxx,maxy,minx,miny都是mxn大小的数组,maxx[i,j]和maxy[i,j]表示了fir_array中的第i个bbox和sec_array中的第j个bbox
    相交部分的左上角的坐标,minx[i,j]和miny[i,j]则表示了右下角的坐标"""
    maxx = np.maximum(fir_array[:, [0]], sec_array[:, [0]].transpose())
    maxy = np.maximum(fir_array[:, [1]], sec_array[:, [1]].transpose())
    minx = np.minimum(fir_array[:, [2]], sec_array[:, [2]].transpose())
    miny = np.minimum(fir_array[:, [3]], sec_array[:, [3]].transpose())

    iw=np.clip((minx-maxx),0.0,None)
    ih=np.clip((miny-maxy),0.0,None)
    iarea=iw*ih #这里的iarea也是一个mxn的数组,iarea[i.j]表示了fir_array中的第i个bbox和sec_array中的第j个bbox的相交部分的面积.

    fir_area=fir_area[:,np.newaxis]
    sec_area=sec_area[np.newaxis,:]
    area=fir_area+sec_area-iarea

    iou=iarea/(area+1)
    return iou

def evalutation(predict_path,gt_path,iou_thresh=0.5):
    """
    评估函数
    :param predict_path: 由预测得到的json文件的目录
    :param gt_path: 由gt得到的json文件的目录
    :param iou_thresh: iou阈值
    :return: 最终计算得到的map
    """
    rcThres=np.linspace(0.0,1.0,101,endpoint=True) #rcThres是设定的召回率的阈值,这里采取了coco的方式,从0.0, 0.1, 0.2 ..., 1.0, 共101个

    #predict_files是所有预测得到的roi的json文件
    predict_files=os.listdir(predict_path)
    #gt_files是所有gt得到的roi的json文件
    gt_files=os.listdir(gt_path)
    #effective_files中只保留有效的roi的json文件,即roi上有ground truth的bbox
    effective_files=[]
    uneffective_files=[]

    #GT用来统计所有的roi中共有多少gt的bbox
    GT=0

    """筛选出所有有效的roi对应的文件名"""
    for predict_file in predict_files:
        if(predict_file in gt_files):
            effective_files.append(predict_file)
        else:
            uneffective_files.append(predict_file)

    #dtms[i]是一个1xn的数组,其中n表示第i个roi所对应的bbox预测值的数量, 如果第i个roi的第j个bbox预测值匹配到了某个bbox的真实值, 则dtms[i][j]
    #对应了这个bbox真实值的id,否则dtms[i][j]为0
    dtms=[]
    #scores[i]是一个1xn的数组,其中n表示第i个roi所对应的bbox预测值的数量,scores[i][j]表示第i个roi的第j个bbox预测值的得分
    scores=[]
    for file in effective_files:
        """下面是读取预测的json文件和真实的json文件"""
        p_path=os.path.join(predict_path,file)
        g_path=os.path.join(gt_path,file)
        p=open(p_path)
        g=open(g_path)
        p_json=json.load(p)
        g_json=json.load(g)
        p.close()
        g.close()

        #predict里面保存这个roi所有的bbox预测值
        predict=np.zeros([0,4],dtype=np.float)
        #score是这个roi所有的bbox预测值的置信度
        score=np.zeros([0,],dtype=np.float)
        for i in range(len(p_json)):
            cur_predict=np.zeros([1,4],dtype=np.float)
            cur_predict[0, 0] = p_json[i]["x"]
            cur_predict[0, 1] = p_json[i]["y"]
            cur_predict[0, 2] = p_json[i]["x"] + p_json[i]["w"]
            cur_predict[0, 3] = p_json[i]["y"] + p_json[i]["h"]
            score=np.append(score,p_json[i]["p"])
            predict=np.append(predict,cur_predict,axis=0)
        #gt是这个roi所有的bbox真实值
        gt = np.zeros([0, 4], dtype=np.float)
        for i in range(len(g_json)):
            cur_gt = np.zeros([1, 4], dtype=np.float)
            cur_gt[0, 0] = g_json[i]["x"]
            cur_gt[0, 1] = g_json[i]["y"]
            cur_gt[0, 2] = g_json[i]["x"] + g_json[i]["w"]
            cur_gt[0, 3] = g_json[i]["y"] + g_json[i]["h"]
            gt = np.append(gt, cur_gt, axis=0)

        """下面是根据置信度从大到小的顺序对该roi的bbox预测值进行排序"""
        index=np.argsort(-score)
        predict=predict[index,:]
        score=score[index]
        """下面就是计算iou数组了"""
        ious=compute_iou(predict,gt)

        dtm=np.zeros([len(predict),])
        gtm=np.zeros([len(gt),])
        for pind, p_ in enumerate(predict):
            iou=iou_thresh
            m=-1
            for gind,g_ in enumerate(gt):
                if(gtm[gind]>0):
                    continue
                if(ious[pind,gind]<iou):
                    continue
                iou=ious[pind,gind]
                m=gind
            if(m==-1):
                continue
            gtm[m]=pind+1
            dtm[pind]=m+1
        dtms.append(dtm)
        GT+=len(gt)
        scores.append(score)

    for file in uneffective_files:
        p_path = os.path.join(predict_path, file)
        p = open(p_path)
        p_json = json.load(p)
        p.close()

        # predict里面保存这个roi所有的bbox预测值
        predict = np.zeros([0, 4], dtype=np.float)
        # score是这个roi所有的bbox预测值的置信度
        score = np.zeros([0, ], dtype=np.float)
        for i in range(len(p_json)):
            cur_predict = np.zeros([1, 4], dtype=np.float)
            cur_predict[0, 0] = p_json[i]["x"]
            cur_predict[0, 1] = p_json[i]["y"]
            cur_predict[0, 2] = p_json[i]["x"] + p_json[i]["w"]
            cur_predict[0, 3] = p_json[i]["y"] + p_json[i]["h"]
            score = np.append(score, p_json[i]["p"])
            predict = np.append(predict, cur_predict, axis=0)

        index = np.argsort(-score)
        predict = predict[index, :]
        score = score[index]

        dtm = np.zeros([len(predict), ])
        dtms.append(dtm)
        scores.append(score)


    dtms=np.concatenate(dtms)
    scores=np.concatenate(scores)
    inds=np.argsort(-scores)
    dtms=dtms[inds]
    tps=(dtms!=0).astype(np.float)
    fps=(dtms==0).astype(np.float)
    tp_sum=np.cumsum(tps)
    fp_sum=np.cumsum(fps)

    RC=fp_sum/GT
    PR=tp_sum/(tp_sum+fp_sum)

    for i in range(len(PR)-1,0,-1):
        if(PR[i]>PR[i-1]):
            PR[i-1]=PR[i]
    inds=np.searchsorted(RC,rcThres,side="left")

    PR=PR[inds]

    return np.mean(PR)

if __name__ == "__main__":
    # prepare_gt_data(cfg.train_path)
    # a=np.array([[100,100,200,200],[150,150,300,250]])
    # b=np.array([[150,150,250,250],[200,200,400,400]])
    map1=evalutation(cfg.eval_predict_data_path,cfg.eval_gt_data_path,0.5)
    map2=evalutation(cfg.eval_predict_data_path,cfg.eval_gt_data_path,0.3)
    map = (map1+map2)/2
    print("%.4f"%map)