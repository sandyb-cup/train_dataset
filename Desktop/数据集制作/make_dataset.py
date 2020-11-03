import h5py
import numpy as np
import scipy
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image
import os
import time, math 

pict_file_path = "myface_photograph/"
pict_file_path2 = "ont_myface/"

# add my face information
def image_format_array(w, h, c):
    myface_train_list = []
    not_myface_train_list = []
    for pict in (os.listdir(pict_file_path)):
        pict_path = pict_file_path+str(pict)
        
        if pict_path != pict_file_path+".DS_Store":
            read_pict = np.array(plt.imread(pict_path))
            read_reshape = scipy.misc.imresize(read_pict,(w, h, c))
            myface_train_list.append(read_reshape)
            
    myface_train_y = np.ones((1, len(myface_train_list)))
    myface_train_x = np.array(myface_train_list)
    n = 0 
    for pict2 in (os.listdir(pict_file_path2)):
        pict_path_merge = pict_file_path2 + str(pict2)
        if pict_path_merge != pict_file_path2 + ".DS_Store":
            read_pict2 = np.array(plt.imread(pict_path_merge))

            if read_pict2.shape[2] == 3: 
                n += 1   
                reshape_array = scipy.misc.imresize(read_pict2, (w, h, c))
                not_myface_train_list.append(reshape_array)
            else:
                continue
            
    not_myface_train_x = np.array(not_myface_train_list) # len(not_myface_train_list) = 1040
    not_myface_train_y = np.zeros((1, len(not_myface_train_x)))
    print("not_myface_train.shape",not_myface_train_x.shape)
    
    
    return myface_train_x, myface_train_y, not_myface_train_x, not_myface_train_y

# Add other face information 
def load_dataset():
    train_dataset = h5py.File("datasets/train_happy.h5", "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # (600, 64, 64, 3)
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    
    # 将插进来的图片转换成 （224，224，3）
    x_train_list = []
    for tr in range(train_set_x_orig.shape[0]):
        train_set_x_orig_tran = scipy.misc.imresize(train_set_x_orig[tr],size = (224, 224, 3))
        x_train_list.append(train_set_x_orig_tran)
    x_train_dataset = np.array(x_train_list)
    
    test_dataset = h5py.File("datasets/test_happy.h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])
    
    x_test_list = []
    for te in range(test_set_x_orig.shape[0]):
        test_set_x_orig_tran = scipy.misc.imresize(test_set_x_orig[te], size = (224, 224, 3))
        x_test_list.append(test_set_x_orig_tran)
    x_test_dataset = np.array(x_test_list)
    
    classes = np.array(test_dataset["list_classes"][:])
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return x_train_dataset, train_set_y_orig, x_test_dataset, test_set_y_orig, classes

# Train and test dataset settle
def dataset_settle(train_dataset_num, test_dataset_num):
    np.random.seed(1)
    
    myface_train_x, myface_train_y, not_myface_train_x, not_myface_train_y = image_format_array(224, 224, 3) #  加载自己制作的数据集 myface_train_x(303, 224, 224, 3) ont_myface_train_x (1036,224,224,4)
    train_set_x_orig, train_set_y_orig, test_set_x_orig,  test_set_y_orig, classes = load_dataset()
    # myface_train_x (48, 224, 224, 3), myface_train_y (1, 48), not_myface_train_x(1036, 224, 224, 3), not_myface_train(1, 1036)
    
    # Shuffle dataset
    # 根据数据集对插入上来的数据集进行随机打乱
    collect_num = np.random.randint(train_dataset_num, size = train_dataset_num) 
    other_dataset_collect_x = train_set_x_orig[:364,:,:,:]
    other_dataset_collect_y = np.zeros(364).reshape(1, 364) # 这里是为了给最后的数据集凑一个整数1200
    
    # 制作训练集
    for index in range(myface_train_x.shape[0]):
        other_dataset_collect_x[index, :, :, :] = myface_train_x[index, :, :, :]
        other_dataset_collect_y[:, index] = myface_train_y[:, index] # 原来的(64, 64, 3)的数据集跟my_face数据集合成在一起
    
    other_dataset_collect_x = np.concatenate((other_dataset_collect_x, not_myface_train_x[:336,:,:,:]), axis = 0) # 对输入的数据集进行剪切
    other_dataset_collect_y = np.concatenate((other_dataset_collect_y, not_myface_train_y), axis = 1)

    # Rrandom train dataset
    random_list = np.random.randint(train_dataset_num, size = train_dataset_num) #  1200
    train_x_orig = other_dataset_collect_x[random_list, :, :, :]
    train_y_orig = other_dataset_collect_y[:, random_list]
    
    # Make test dataset 150
    test_random_list = np.random.randint(test_dataset_num, size = test_dataset_num)
    
    test_y_zeros = np.zeros(test_dataset_num).reshape(1, test_dataset_num)
    test_lable_ones = np.ones(myface_train_x.shape[0]).reshape(1, myface_train_x.shape[0])
    
    # 进行数据替换
    for i in range(test_dataset_num):
        test_set_x_orig[i,:, :, :,] = myface_train_x[i, :, :, :]
        test_y_zeros[:, i] = test_lable_ones[:, i]
        
    test_x_orig = test_set_x_orig[test_random_list,:, :, :]
    test_y_orig = test_y_zeros[:, test_random_list]
    
    # Make train h5 file
    file_name = "my_dataset/"
    os.mkdir(file_name)
    file = h5py.File(file_name+"train_dataset",'w')
    
    g1 = file.create_group("my_train_set")
    d1 = g1.create_dataset("train_x", data = train_x_orig)
    d2 = g1.create_dataset('train_y', data = train_y_orig)
    
    file2 = h5py.File(file_name + "test_dataset", 'w')
    g2 = file2.create_group("my_test_set")
    d3 = g2.create_dataset("test_x", data = test_x_orig)
    d4 = g2.create_dataset('test_y', data = test_y_orig)
                        
    return train_x_orig, train_y_orig, test_x_orig, test_y_orig,
    
def main():
 
    train_x_orig, train_y_orig, test_x_orig, test_y_orig = dataset_settle(700, 150) # 训练集张数 测试集张数
    print(train_x_orig.shape)
    print(test_x_orig.shape)
    
if __name__=="__main__":
    main()

