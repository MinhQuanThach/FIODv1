from types import SimpleNamespace

def get_arguments():
    args = SimpleNamespace()
    args.batch_size = 4
    args.iter_size = 1
    args.num_workers = 2
    args.data_dir = '/kaggle/input/fiod-full/FIOD_dataset/data'
    args.img_size = 640
    args.num_classes = 8  # Adjust based on your dataset
    args.num_steps = 1000
    args.random_seed = 1234
    args.snapshot_dir = '/content/drive/MyDrive/FIOD_dataset/snapshots'
    args.set = 'train'
    args.lambda_fsm = 100  # Fog style matching weight
    args.lambda_con = 1  # Consistency loss weight
    args.file_name = 'yolov8n_fifo'
    args.gpu = 0
    return args