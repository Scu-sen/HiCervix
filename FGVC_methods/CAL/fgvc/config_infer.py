##################################################
# Training Config
##################################################
GPU = '5'                   # GPU
workers = 4                 # number of Dataloader workers
epochs = 160                # number of epochs
batch_size = 64            # batch size
learning_rate = 1e-3        # initial learning rate

##################################################
# Model Config
##################################################
image_size = (448, 448)     # size of training images
net = 'resnet101'  # feature extractor
num_attentions = 32         # number of attention maps
beta = 5e-2                 # param for update feature centers

visual_path = None  # './vis-cub-inception-cf/'  # None

##################################################
# Dataset/Path Config
##################################################
tag = 'tct'                # 'aircraft', 'bird', 'car', or 'dog'

# checkpoint model for resume training
ckpt = './FGVC/tct/wsdan-resnet101-cal/model_bestacc.pth'
test_csv = 'dataset/hierarchy_classification/version2023/test_image_path_keep_species_all.csv'
is_test = True
