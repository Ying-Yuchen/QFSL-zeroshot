import os


class MODEL:
    img_size = 224
    ori_size = 256
    weights_decay =  0.0005

class Config:
    Model = MODEL
    seed = 0
    score_dic_path = os.path.join('checker',"logger")
    checkpoints_dir = os.path.join('checker',"checkpoints")
    test_score_path = os.path.join(score_dic_path,"test_results.txt")
    train_score_path = os.path.join(score_dic_path,"train_results.txt")


