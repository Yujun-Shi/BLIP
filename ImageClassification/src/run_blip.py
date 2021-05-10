import math
import sys,os,argparse,time
import numpy as np
import torch
import torch.nn as nn
import utils
from datetime import datetime
from networks.quant_layer import Linear_Q, Conv2d_Q
from approaches.blip_utils import estimate_fisher


def main():
    tstart=time.time()

    parser=argparse.ArgumentParser(description='BLIP Image Classification')

    # Data parameters
    parser.add_argument('--seed',               default=0,                   type=int,     help='(default=%(default)d)')
    parser.add_argument('--device',             default='cuda:0',            type=str,     help='gpu id')
    parser.add_argument('--experiment',         default='mnist5',            type=str,     help='experiment dataset', required=True)
    parser.add_argument('--data_path',          default='../data/',          type=str,     help='gpu id')

    # Training parameters
    parser.add_argument('--approach',           default='blip',              type=str,     help='continual learning approach')
    parser.add_argument('--output',             default='',                  type=str,     help='')
    parser.add_argument('--checkpoint_dir',     default='../checkpoints/',   type=str,     help='')
    parser.add_argument('--nepochs',            default=200,                 type=int,     help='')
    parser.add_argument('--sbatch',             default=64,                  type=int,     help='')
    parser.add_argument('--lr',                 default=0.05,                type=float,   help='')
    parser.add_argument('--momentum',           default=0,                   type=float,   help='')
    parser.add_argument('--weight-decay',       default=0.0,                 type=float,   help='')
    parser.add_argument('--resume',             default='no',                type=str,     help='resume?')
    parser.add_argument('--sti',                default=0,                   type=int,     help='starting task?')

    # Model parameters
    parser.add_argument('--ndim',               default=1200,                type=int,     help='hidden dimension for 2 layer MLP')
    parser.add_argument('--mul',                default=1.0,                 type=float,   help='multiplier of model width')

    # BLIP specific parameters
    parser.add_argument('--max-bit',            default=20,                  type=int,     help='maximum number of bits for each parameter')
    parser.add_argument('--F-prior',            default=1e-15,               type=float,   help='scaling factor of F_prior')

    args=parser.parse_args()
    utils.print_arguments(args)

    #####################################################################################

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print('Using device:', args.device)
    checkpoint = utils.make_directories(args)
    args.checkpoint = checkpoint
    print()

    # Args -- Experiment
    if args.experiment=='mnist2':
        from dataloaders import mnist2 as dataloader
    elif args.experiment=='mnist5':
        from dataloaders import mnist5 as dataloader
    elif args.experiment=='pmnist':
        from dataloaders import pmnist as dataloader
    elif args.experiment=='cifar':
        from dataloaders import cifar as dataloader
    elif args.experiment=='mixture5':
        from dataloaders import mixture5 as dataloader
    else:
        raise NotImplementedError('dataset currently not implemented')

    # Args -- Approach
    if args.approach=='blip':
        from approaches import blip as approach
    else:
        raise NotImplementedError('approach currently not implemented')

    # Args -- Network
    if args.experiment=='mnist2' or args.experiment=='pmnist' or args.experiment == 'mnist5':
        from networks import q_mlp as network
    else:
        from networks import q_alexnet as network


    ########################################################################################
    print()
    print("Starting this run on :")
    print(datetime.now().strftime("%Y-%m-%d %H:%M"))

    # Load
    print('Load data...')
    data,taskcla,inputsize=dataloader.get(data_path=args.data_path, seed=args.seed)
    print('Input size =',inputsize,'\nTask info =',taskcla)
    args.num_tasks=len(taskcla)
    args.inputsize, args.taskcla = inputsize, taskcla

    # Inits
    print('Inits...')
    model=network.Net(args).to(args.device)

    print('-'*100)
    appr=approach.Appr(model,args=args)
    print('-'*100)

    if args.resume == 'yes':
        checkpoint = torch.load(os.path.join(args.checkpoint, 'model_{}.pth.tar'.format(args.sti)))
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device=args.device)
    else:
        args.sti = 0

    # Loop tasks
    acc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
    lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
    num_task = len(taskcla)
    for t,ncla in taskcla[args.sti:]:

        print('*'*100)
        print('Task {:2d} ({:s})'.format(t,data[t]['name']))
        print('*'*100)

        # Get data
        xtrain=data[t]['train']['x'].to(args.device)
        ytrain=data[t]['train']['y'].to(args.device)
        xvalid=data[t]['valid']['x'].to(args.device)
        yvalid=data[t]['valid']['y'].to(args.device)
        task=t

        # Train
        appr.train(task,xtrain,ytrain,xvalid,yvalid)
        print('-'*100)

        # BLIP specifics post processing
        estimate_fisher(task, args.device, model, xtrain, ytrain)
        for m in model.features.modules():
            if isinstance(m, Linear_Q) or isinstance(m, Conv2d_Q):
                # update bits according to information gain
                m.update_bits(task=task, C=0.5/math.log(2))
                # do quantization
                m.sync_weight()
                # update Fisher in the buffer
                m.update_fisher(task=task)

        # save the model after the update
        appr.save_model(task)
        # Test
        for u in range(t+1):
            xtest=data[u]['test']['x'].to(args.device)
            ytest=data[u]['test']['y'].to(args.device)
            test_loss,test_acc=appr.eval(u,xtest,ytest,debug=True)
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.3f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
            acc[t,u]=test_acc
            lss[t,u]=test_loss

        utils.used_capacity(model, args.max_bit)

        # Save
        print('Save at '+args.checkpoint)
        np.savetxt(os.path.join(args.checkpoint,'{}_{}_{}.txt'.format(args.experiment,args.approach,args.seed)),acc,'%.5f')

    utils.print_log_acc_bwt(args, acc, lss)
    print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))


if __name__ == '__main__':
    main()
