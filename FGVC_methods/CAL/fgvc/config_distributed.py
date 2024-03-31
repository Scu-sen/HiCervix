##################################################
# Training Config
##################################################
workers = 16                 # number of Dataloader workers
epochs = 50                # number of epochs 160
batch_size = 10             # batch size
learning_rate = 1e-3        # initial learning rate

##################################################
# Model Config
##################################################
net = 'resnet101'  # feature extractor
num_attentions = 32     # number of attention maps
beta = 5e-2                 # param for update feature centers

##################################################
# Dataset/Path Config
##################################################
# tag = 'bird'                # 'aircraft', 'bird', 'car', or 'dog'
tag = 'tct'                # 'aircraft', 'bird', 'car', or 'dog'

train_csv = 'dataset/hierarchy_classification/version2023/train_image_path_keep_species.csv'
val_csv = 'dataset/hierarchy_classification/version2023/val_image_path_keep_species.csv'

# saving directory of .ckpt models
# save_dir = './FGVC/bird/wsdan-resnet101-cal/'
save_dir = './FGVC/tct/wsdan-resnet101-cal/'
model_name = 'model.ckpt'
log_name = 'train.log'

# checkpoint model for resume training
ckpt = False
# ckpt = save_dir + model_name