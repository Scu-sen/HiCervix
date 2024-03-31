import torch.cuda
import torch.nn
from torchvision import models
import timm

def init_model_on_gpu(gpus_per_node, opts):
    arch_dict = models.__dict__
    pretrained = False if not hasattr(opts, "pretrained") else opts.pretrained
    distributed = False if not hasattr(opts, "distributed") else opts.distributed
    print("=> using model '{}', pretrained={}".format(opts.arch, pretrained))
    if opts.arch == "swinT":
        model = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=False)
        checkpoint = torch.load('classification_hierarchy/making-better-mistakes-swinT/swinT_L_best_from_PIM.pt')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint              # pretrained with imagenet
        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                state_dict[k[len("backbone."):]] = state_dict[k] 
            del state_dict[k] # delete renamed or unused k
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print("Missing keys")
        print(msg.missing_keys)
        print("Unexpected keys")
        print(msg.unexpected_keys)
    else:
        model = arch_dict[opts.arch](pretrained=pretrained)
    
    
    if opts.arch == "resnet18":
        feature_dim = 512
    elif opts.arch == "resnet50":
        feature_dim = 2048
    elif opts.arch == "swinT":
        feature_dim = 1536
    else:
        ValueError("Unknown architecture ", opts.arch)
    if opts.devise or opts.barzdenzler:
        if opts.pretrained or opts.pretrained_folder:
            for param in model.parameters():
                if opts.train_backbone_after == 0:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if opts.use_2fc:
            if opts.use_fc_batchnorm:
                model.fc = torch.nn.Sequential(
                    torch.nn.Linear(in_features=feature_dim, out_features=opts.fc_inner_dim, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(opts.fc_inner_dim),
                    torch.nn.Linear(in_features=opts.fc_inner_dim, out_features=opts.embedding_size, bias=True),
                )
            else:
                model.fc = torch.nn.Sequential(
                    torch.nn.Linear(in_features=feature_dim, out_features=opts.fc_inner_dim, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.Linear(in_features=opts.fc_inner_dim, out_features=opts.embedding_size, bias=True),
                )
        else:
            if opts.use_fc_batchnorm:
                model.fc = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(feature_dim), torch.nn.Linear(in_features=feature_dim, out_features=opts.embedding_size, bias=True)
                )
            else:
                model.fc = torch.nn.Sequential(torch.nn.Linear(in_features=feature_dim, out_features=opts.embedding_size, bias=True))
    else:
        #model.fc = torch.nn.Sequential(torch.nn.Dropout(opts.dropout), torch.nn.Linear(in_features=feature_dim, out_features=opts.num_classes, bias=True))
        model.head =torch.nn.Linear(in_features=feature_dim, out_features=opts.num_classes)
    #print(model.fc)
    if distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opts.gpu is not None:
            torch.cuda.set_device(opts.gpu)
            model.cuda(opts.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            opts.batch_size = int(opts.batch_size / gpus_per_node)
            opts.workers = int(opts.workers / gpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opts.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif opts.gpu is not None:
        torch.cuda.set_device(opts.gpu)
        model = model.cuda(opts.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    return model
