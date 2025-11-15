# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import umap
import autoencoder


# %%
faceRec_dataPath = 'FaceRecognition'
carDet_dataPath = 'CarDetection'

# %%
def load_data(root,dirName):
    data_path = os.path.join(root,dirName)
    data_dir = sorted(os.listdir(data_path))
    img_data = []
    labels = []
    for img_name in data_dir:
        img_path = os.path.join(data_path,img_name)
        img = cv.imread(img_path)
        labels.append(int(img_name.split('_')[0]))
        img_data.append(img)
    
    return np.array(labels), np.array(img_data)


# %%
faceRec_labels,faceRec_img_data = load_data(faceRec_dataPath,'train')

# %%
def PCA(X,p):
    # X = [x.flatten() for x in X]
    # X = np.array(X)
    # X = X/(np.linalg.norm(X,axis=1,keepdims=True))
    X_m = np.mean(X,axis=0)
    X = X - X_m
    X = X.T
    cov_mat = (X.T)@X
    u,d,v = np.linalg.svd(cov_mat)
    W = X@v
    W = W/(np.linalg.norm(W,axis=0,keepdims=True))
    return W[:,:p],X_m
    


# %%

# aaa = PCA(faceRec_img_data,p=1)

# %%
def LDA(X,Y,p):
    # X = [x.flatten() for x in X]
    # X = np.array(X)
    # X = X/(np.linalg.norm(X,axis=1,keepdims=True))
    X_m = np.mean(X,axis=0)
    X = X - X_m
    num_classes = len(np.unique(Y))
    class_means = []
    data_mean_diff = []
    for label in np.unique(Y):
        X_c = X[Y==label]
        class_mean = np.mean(X_c,axis=0)
        class_means.append(class_mean)
        X[Y==label] = (X_c - class_mean)
    # X -= X_m
    class_means = np.array(class_means)
    mean_vec_diff = (class_means-X_m).T
    S_B = mean_vec_diff.T@(mean_vec_diff)
    u,d,v = np.linalg.svd(S_B)
    W = (mean_vec_diff)@v
    W = W/(np.linalg.norm(W,axis=0,keepdims=True))
    D_B = d[:-1]
    Y = W[:,:-1]
    Z = Y@(np.sqrt(np.linalg.inv(np.diag(D_B))))
    zt_Sw_z = ((X@Z).T)@(X@Z) #(Z.T)@X.T@X@Z
    u,d,v = np.linalg.svd(zt_Sw_z)
    # v = ((X@Z))@vt  #
    # v = v/(np.linalg.norm(v,axis=0,keepdims=True))  #
    # print((X@Z).shape,v.shape,Z.shape)
    W = Z@v[:,1:]#Z@v[:,1:]
    W = W/(np.linalg.norm(W,axis=0,keepdims=True))
    # print(W[:,-p:].shape)
    return W[:,-p:],X_m  #W[:,-p:]
    

# %%
# ##########################################################
# faceRec_img_data = [x.flatten() for x in faceRec_img_data]
# faceRec_img_data = np.array(faceRec_img_data)
# faceRec_img_data = faceRec_img_data/(np.linalg.norm(faceRec_img_data,axis=1,keepdims=True))
# aaa=LDA(faceRec_img_data,faceRec_labels,p=1)

# %%
def NearestNeighbor(test_vec,feature_space,y):
    distances = np.linalg.norm(feature_space-test_vec,axis=1)
    nearest_idx = np.argmin(distances)
    y_pred = y[nearest_idx]
    return y_pred


# %%
def classifer(x_train,y_train,x_test,y_test,p,classifier_type='PCA',UMAP_plot=False):
    if classifier_type!='autoencoder':
        if classifier_type == 'PCA':
            l_dim_space,X_m = PCA(x_train,p)
            # x_train -= X_m
            # x_test -= X_m
        elif classifier_type == 'LDA':
            l_dim_space,X_m = LDA(x_train,y_train,p)

        x_train -= X_m
        x_test -= X_m

        x_train_proj = x_train@l_dim_space
        x_test_proj = x_test@l_dim_space
    
    elif classifier_type=='autoencoder':
        x_train_proj = x_train
        x_test_proj = x_test

    y_pred = []
    for test_vec in x_test_proj:
        y_pred.append(NearestNeighbor(test_vec,x_train_proj,y_train))
    y_pred = np.array(y_pred)
    accuracy = np.sum(y_pred==y_test)/len(y_test)
    print(f'Accuracy with p={p} : ',accuracy)

    if UMAP_plot:
        # train_reducer = umap.UMAP()
        # train_embedding = train_reducer.fit_transform(x_train_proj)
        # test_reducer = umap.UMAP()
        # test_embedding = test_reducer.fit_transform(x_test_proj)
        reducer = umap.UMAP()
        train_embedding = reducer.fit_transform(x_train_proj)
        test_embedding = reducer.transform(x_test_proj)
        fig,axes = plt.subplots(1,2)
        plt.suptitle(f'UMAP plot for p={p}')
        axes[0].scatter(train_embedding[:,0],train_embedding[:,1],c=y_train)
        axes[0].set_title('train data')
        axes[1].scatter(test_embedding[:,0],test_embedding[:,1],c=y_pred)
        axes[1].set_title('test data')
        plt.show()

    return accuracy



# %%
def faceRec_classification(faceRec_dataPath):
    y_train,x_train = load_data(faceRec_dataPath,'train')
    y_test,x_test = load_data(faceRec_dataPath,'test')

    x_train = [x.flatten() for x in x_train]
    x_train = np.array(x_train)
    x_train = x_train/(np.linalg.norm(x_train,axis=1,keepdims=True))
    x_test = [x.flatten() for x in x_test]
    x_test = np.array(x_test)
    x_test = x_test/(np.linalg.norm(x_test,axis=1,keepdims=True))

    p_list = [1,3,5,7,8,9,11,16]
    PCA_acc_list = []
    LDA_acc_list = []
    autoencoder_acc_list = []
    print('PCA')
    for p in p_list:
        PCA_acc = classifer(x_train,y_train,x_test,y_test,p,classifier_type='PCA',UMAP_plot=True)
        PCA_acc_list.append(PCA_acc)
    
    print('LDA')
    for p in p_list:
        LDA_acc = classifer(x_train,y_train,x_test,y_test,p,classifier_type='LDA',UMAP_plot=True)
        LDA_acc_list.append(LDA_acc)
    
    print('Autoencoder')
    for p in [3, 8, 16]:
        X_train,Y_train,X_test,Y_test = autoencoder.autoencoder_model(p)
        autoencoder_acc = classifer(X_train,Y_train,X_test,Y_test,p,classifier_type='autoencoder',UMAP_plot=True)
        autoencoder_acc_list.append(autoencoder_acc)

    plt.plot(p_list,PCA_acc_list,marker='*',label='PCA')
    plt.plot(p_list,LDA_acc_list,marker='o',label='LDA')
    plt.plot([3, 8, 16],autoencoder_acc_list,marker='x',label='Auoencoder')
    plt.title('Accuracy plot wrt p')
    plt.xlabel('p (feature dim.)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    # print(PCA_acc_list)

# %%
faceRec_classification(faceRec_dataPath)

# %%



