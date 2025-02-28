import torch
import torch.nn as nn
import torch.nn.functional as F


def Regularization_LabelSpaceSparsity(features, targets):
   # outputs (b, feature)
   # targets (b, )
   output = torch.unique(targets)
   regularization = 0

   for i in range(output.shape[0]):
       # divided the targets per label
       extracted = torch.where( targets==output[i] )[0]

       # same label to be -1, different label to be 1
       current_mask = torch.ones(targets.shape).cuda()
       current_mask[extracted] -= 2

       # find a median vector per label and generate median target
       extracted_features = features[extracted,:]
       extracted_features_median = torch.median(extracted_features, 0)[0]
       extracted_features_median = extracted_features_median.repeat(targets.shape[0],1)

       # calculate a cosine similarity
       cosine_similarity_between_labels = F.cosine_similarity(extracted_features_median,features)

       # calculate a similarity and make it in 0~2
       regularization += torch.sum(1+(current_mask * cosine_similarity_between_labels))
      
   return regularization



def Regularization_LabelSpaceFocusing(features, targets):
   # outputs (b, feature)
   # targets (b, )
   output = torch.unique(targets)
   regularization = 0

   for i in range(output.shape[0]):
       # divided the targets per label
       extracted = torch.where( targets==output[i] )[0]

       # same label to be -1
       current_mask = torch.zeros(targets.shape).cuda()
       current_mask[extracted] -= 1

       # find a median vector per label and generate median target
       extracted_features = features[extracted,:]
       extracted_features_median = torch.median(extracted_features, 0)[0]
       extracted_features_median = extracted_features_median.repeat(targets.shape[0],1)

       # calculate a cosine similarity
       cosine_similarity_between_labels = F.cosine_similarity(extracted_features_median,features)

       # calculate a similarity and make it in 0~2
       regularization += torch.sum(1+(current_mask * cosine_similarity_between_labels))
      
   return regularization


