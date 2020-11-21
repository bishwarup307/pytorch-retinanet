class Config:

    # dataset config
    dataset = "coco"
    image_dir = "/home/bishwarup/EV/v0.2.2/data/images"
    val_image_dir = None  # if validation images are stored elsewhere
    train_json_path = "/home/bishwarup/EV/v0.2.2/data/annotations/train_non_empty.json"
    val_json_path = "/home/bishwarup/EV/v0.2.2/data/annotations/val_non_empty.json"
    negative_sampling_rate = None  # sample images with no annotations at batch level
    normalize = dict()
    normalize["mean"] = [0.485, 0.456, 0.406]
    normalize["std"] = [0.229, 0.224, 0.225]
    logdir = "/home/bishwarup/EV/v0.2.2/runs/"

    # aug config
    augs = dict()
    augs["hflip"] = 0.5  # horizontal flip, either False or a probability
    augs["vflip"] = False  # vertical flip, either False or a probability
    augs["color_jitter"] = 0.3  # Color jitter, either False or a probability
    augs["brightness"] = False  # Brightness adjustment, either False or a probability
    augs["contrast"] = False  # Contrast adjustment, either False or a probability
    augs["shiftscalerotate"] = 0.5  # translation, either False or a probability
    augs["gamma"] = False  # gamma correction, either False or a probability
    augs[
        "rgb_shift"
    ] = False  # rgb shift, either False or a tuple (r_shift, g_shift, b_shift, proba)
    augs["sharpness"] = 0.5  # shaprness adjustment, either False or a probability
    augs["perspective"] = False  # perspective transformation, either False or a probability
    augs[
        "cutout"
    ] = False  # random cutout, either False or a tuple (proba, max_h_cutout, max_w_cutout)
    augs["gaussian_blur"] = 0.3  # Gaussian blur, either False or a probability
    augs["superpixels"] = False  # Superpixels, either False or a probability
    augs["additive_noise"] = 0.3  # Gaussian Additive Noise, either False or a probability
    augs["min_visibility"] = 0.8
    augs["min_area"] = 450

    # model config
    backbone = "resnet-50"
    pretrained = False
    freeze_bn = False
    weights = None
    image_size = 512
    alpha = 0.25  # focal loss param
    gamma = 2.0  # focal loss param
    nms_iou_threshold = 0.5
    conf_threshold = 0.1

    # learning config
    num_epochs = 100
    batch_size = 16
    workers = 8
    optimizer = "adam"
    lr_schedule = "WarmupCosineAnnealing"
    base_lr = 1e-4
    final_lr = 0
    weight_decay = 1e-6
    warmup_epochs = 0
    start_warmup = 0
    early_stopping = -1
