import torch
import numpy as np

from configs import FEATURE_SIZE

class CanonShape:
    def __init__(self,class_name,seed_counts=100,feat_dim=FEATURE_SIZE):
        
        self.class_name = class_name
        self.seed_counts = seed_counts
        self.feat_dim = feat_dim

        self.conf_list= []
        self.feat_list= []
        self.pos_list= []

        self.ctr= 0

        self.canon_pos=None
        self.canon_feat=None

    def add_data(self, feat,pos,conf):
        self.feat_list.append(feat)
        self.conf_list.append(conf)
        self.pos_list.append(pos)
        self.ctr += 1
    
    def get_shape(self):
        if torch.is_tensor( self.canon_feat):
            return self.canon_feat,self.canon_pos
        return self.calculate_shape()

    def reset_shape(self):

        self.ctr = 0

        self.canon_pos=None
        self.canon_feat=None

        self.conf_list= []
        self.feat_list= []
        self.pos_list= []


    def calculate_shape(self):
        if self.ctr==0:
            print('no query data to calculate')
            return False
        
        feat_t = torch.stack([self.feat_list],0)
        conf_t = torch.stack([self.conf_list],0)
        pos_t = torch.stack([self.pos_list],0)
        
        self.canon_pos = torch.mean(pos_t*conf_t,0)
        self.canon_feat = torch.mean(feat_t*conf_t,0)

        return self.canon_feat,self.canon_pos

        


