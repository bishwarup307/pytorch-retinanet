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
    logdir = "/home/bishwarup/Desktop/sample_log/"

    # aug config
    augs = dict()
    augs["hflip"] = False  # horizontal flip, either False or a probability
    augs["vflip"] = False  # vertical flip, either False or a probability
    augs["color_jitter"] = False  # Color jitter, either False or a probability
    augs["brightness"] = False  # Brightness adjustment, either False or a probability
    augs["contrast"] = False  # Contrast adjustment, either False or a probability
    augs["shiftscalerotate"] = False  # translation, either False or a probability
    augs["gamma"] = False  # gamma correction, either False or a probability
    augs[
        "rgb_shift"
    ] = False  # rgb shift, either False or a tuple (r_shift, g_shift, b_shift, proba)
    augs["sharpness"] = False  # shaprness adjustment, either False or a probability
    augs[
        "perspective"
    ] = False  # perspective transformation, either False or a probability
    augs[
        "cutout"
    ] = False  # random cutout, either False or a tuple (proba, max_h_cutout, max_w_cutout)
    augs["gaussian_blur"] = False  # Gaussian blur, either False or a probability
    augs["superpixels"] = False  # Superpixels, either False or a probability
    augs[
        "additive_noise"
    ] = False  # Gaussian Additive Noise, either False or a probability
    augs["min_visibility"] = 0.8
    augs["min_area"] = 450

    # model config
    backbone = "resnet-18"
    pretrained = True
    weights = None
    image_size = 512

    # learning config
    num_epochs = 100
    batch_size = 8
    workers = 0
    optimizer = "adam"
    lr_schedule = "WarmupCosineAnnealing"
    base_lr = 1e-5
    final_lr = 0
    weight_decay = 1e-6
    warmup_epochs = 0
    start_warmup = 0
    early_stopping = 10