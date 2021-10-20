import argparse

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self):
        # ---------------------------------------- step 1/5 : parameters preparing... ----------------------------------------
        self.parser.add_argument("--results_dir", type=str, default='../results', help="path of results")
        self.parser.add_argument("--trial", type=str, default='baseline', help="trial name")
        self.parser.add_argument("--pretrained_dir", type=str, default='../pretrained', help="path of pretrained models")
        self.parser.add_argument("--log_dir", type=str, default='/zhaosuiyi/visualization/CRNet', help="path of saving log file")
        
        # ---------------------------------------- step 2/5 : data loading... ------------------------------------------------
        self.parser.add_argument("--resizeX", type=int, default=128, help="height of image, after resize")
        self.parser.add_argument("--resizeY", type=int, default=224, help="width of image, after resize")
        self.parser.add_argument("--cropX", type=int, default=128, help="height of image, after crop")
        self.parser.add_argument("--cropY", type=int, default=128, help="width of image, after crop")
        
        self.parser.add_argument("--data_source", type=str, default='/zhaosuiyi/datasets/GOPRO', help="path of datasource")
        self.parser.add_argument("--batch_size", type=int, default=2, help="size of the batches when training")
        self.parser.add_argument("--val_size", type=int, default=3, help="size of the batches when validating and saving")
        self.parser.add_argument("--n_cpus", type=int, default=4, help="number of cpu threads to use during batch generation")
        
        # ---------------------------------------- step 3/5 : model defining... ----------------------------------------------
        self.parser.add_argument('--PGBFP', action='store_true', default = False, help="if true, use module 'pyramid global blur feature perception' ")
        self.parser.add_argument("--n_blocks", type=int, default=9, help="number of blocks in generator")
        self.parser.add_argument("--n_scales", type=int, default=3, help="the scale of discriminator, can not be less than 1, if specified as 1, it will performed as a normal gan")
        self.parser.add_argument("--scale_type", type=str, default='crop', help="optioal: resize, crop. type of image scale in multi-scale discrimator")
        
        self.parser.add_argument('--resume', action='store_true', default = False, help='if true, resume execution where it left off; when using it, make sure you have trained the program before')
        self.parser.add_argument('--pretrain', action='store_true', default = False, help='if true, start training with a pre-trained model')
        self.parser.add_argument('--model_name', type=str, default='optimal_params', help='model name, model path consist of results_dir and model_name')
        
        # ---------------------------------------- step 4/5 : requisites defining... -----------------------------------------
        self.parser.add_argument("--gan_type", type=str, default='lsgan', help="the type of gan, optional: lsgan, vanilla, wgan-gp")
        
        self.parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
        self.parser.add_argument("--n_epochs", type=int, default=80, help="number of epochs of training")
        self.parser.add_argument("--decay_epoch", type=int, default=40, help="epoch from which to start lr decay")
        
        # ---------------------------------------- step 5/5 : training... ----------------------------------------------------
        self.parser.add_argument("--start_epoch", type=int, default=0, help="epoch to start training from")
        self.parser.add_argument("--lambda_cycle", type=float, default=20.0, help="cycle loss weight")
        self.parser.add_argument("--lambda_perc", type=float, default=0., help="perceptual loss weight")
        self.parser.add_argument("--lambda_identity", type=float, default=0., help="identity loss weight")
        self.parser.add_argument("--show_gap", type=int, default=10, help="the gap for printing the information of training and record scalar to SummaryWriter, in iteration")
        
        self.parser.add_argument("--val_gap", type=int, default=2, help="the gap between two validations, in epoch")
        self.parser.add_argument("--save_gap", type=int, default=5, help="the gap between two saving model, in epoch, e.g., save the model every save_gap*val_gap epochs")
        self.parser.add_argument('--patience', type=int, default=5, help='number of validating performance of model with no improvement, after which paremeters updating will be stoped')
        
    def parse(self):
        self.initialize()
        opt = self.parser.parse_args()
        
        self.show(opt)
        
        return opt
    
    def show(self,opt):
        
        args = vars(opt)
        print('************ Options ************')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('************** End **************')
        

class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self):
        # ---------------------------------------- step 1/4 : parameters preparing... ----------------------------------------
        self.parser.add_argument("--output_dir", type=str, default='../output', help="path of output")
        self.parser.add_argument("--trial", type=str, default='trial1', help="trial name")
        self.parser.add_argument("--pretrained_dir", type=str, default='../pretrained', help="path of pretrained models")
        
        # ---------------------------------------- step 2/4 : data loading... ------------------------------------------------
        self.parser.add_argument("--resizeX", type=int, default=128, help="height of image, after resize")
        self.parser.add_argument("--resizeY", type=int, default=224, help="width of image, after resize")
        
        self.parser.add_argument("--data_source", type=str, default='/zhaosuiyi/datasets/GOPRO', help="path of datasource")
        self.parser.add_argument("--val_size", type=int, default=1, help="size of the batches when validating and saving")
        self.parser.add_argument("--n_cpus", type=int, default=4, help="number of cpu threads to use during batch generation")
        
        # ---------------------------------------- step 3/4 : model defining... ----------------------------------------------
        self.parser.add_argument('--PGBFP', action='store_true', default = False, help="if true, use module 'pyramid global blur feature perception' ")
        self.parser.add_argument("--n_blocks", type=int, default=9, help="number of blocks in generator")
        
        self.parser.add_argument('--model_name', type=str, default='optimal_params', help='model name, model path consist of results_dir and model_name')
        
        # ---------------------------------------- step 4/4 : testing... ----------------------------------------------------
        
    def parse(self):
        self.initialize()
        opt = self.parser.parse_args()
        
        self.show(opt)
        
        return opt
    
    def show(self,opt):
        
        args = vars(opt)
        print('************ Options ************')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('************** End **************')