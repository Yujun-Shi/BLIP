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

    parser=argparse.ArgumentParser(description='BLIP mini-ImageNet')

    # Data parameters
    parser.add_argument('--seed',               default=0,               type=int,   help='(default=%(default)d)')
    parser.add_argument('--device',             default='cuda:0',        type=str,   help='gpu id')
    parser.add_argument('--experiment',         default='MI',            type=str)
    parser.add_argument('--data_dir',           default='data',          type=str,   help='data directory')
    parser.add_argument('--ntasks',             default=10,              type=int)
    parser.add_argument('--pc-valid',           default=0.02,            type=float)
    parser.add_argument('--workers',            default=4,               type=int)

    # Training parameters
    parser.add_argument('--approach',           default='blip',          type=str)
    parser.add_argument('--output',             default='',              type=str,   help='')
    parser.add_argument('--checkpoint_dir',     default='checkpoints/',  type=str,   help='')
    parser.add_argument('--nepochs',            default=200,             type=int,   help='')
    parser.add_argument('--sbatch',             default=64,              type=int,   help='')
    parser.add_argument('--lr',                 default=0.05,            type=float, help='')
    parser.add_argument('--momentum',           default=0.9,             type=float)
    parser.add_argument('--weight-decay',       default=0.0,             type=float)
    parser.add_argument('--resume',             default='no',            type=str,   help='resume?')
    parser.add_argument('--sti',                default=0,               type=int,   help='starting task?')

    # model parameters
    parser.add_argument('--mul',                default=1.0,             type=float)
    parser.add_argument('--arch',               default='alexnet',       type=str)

    # BLIP parameters
    parser.add_argument('--max-bit',            default=20,              type=int,   help='')
    parser.add_argument('--F-prior',            default=1e-15,           type=float, help='')

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
    from dataloaders.miniimagenet import DatasetGen

    # Args -- Approach
    from approaches import blip as approach

    # Args -- Network
    if args.arch == 'alexnet':
        from networks import q_alexnet as network
    elif args.arch == 'resnet':
        from networks import q_resnet as network
    else:
        raise NotImplementedError("network currently not implemented")

    ########################################################################################
    print()
    print("Starting this run on :")
    print(datetime.now().strftime("%Y-%m-%d %H:%M"))

    # Load
    print('Load data...')
    # prepare data for each task
    datagen = DatasetGen(args)
    for task_id in range(args.ntasks):
        datagen.get(task_id)
    print('\nTask info =',datagen.taskcla)

    args.num_tasks = len(datagen.taskcla)
    args.inputsize, args.taskcla = datagen.inputsize, datagen.taskcla

    # Inits
    print('Inits...')
    model=network.Net(args).to(args.device)

    # print number of parameters
    count = 0
    for p in model.parameters():
        count+=np.prod(p.size())
    print('model size in MB: ', count*4/(1024*1024))

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
    acc=np.zeros((len(args.taskcla),len(args.taskcla)),dtype=np.float32)
    lss=np.zeros((len(args.taskcla),len(args.taskcla)),dtype=np.float32)
    for task, ncla in args.taskcla[args.sti:]:
        data_t = datagen.dataloaders[task]
        print('*'*100)
        print('Task {:2d} ({:s})'.format(task, data_t['name']))
        print('*'*100)

        # Train
        appr.train(task, data_t['train'], data_t['valid'])
        print('-'*100)

        estimate_fisher(task, args.device, model, data_t['fisher'])
        for m in model.modules():
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
        for u in range(task+1):
            data_u = datagen.dataloaders[u]
            test_loss, test_acc=appr.eval(u, data_u['test'])
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.3f}% <<<'.format(u,data_u['name'],test_loss,100*test_acc))
            acc[task,u]=test_acc
            lss[task,u]=test_loss

        utils.used_capacity(model, args.max_bit)

        # Save
        print('Save at '+args.checkpoint)
        np.savetxt(os.path.join(args.checkpoint,'{}_{}_{}_{}.txt'.format(args.approach,args.arch,args.seed,str(args.F_prior))), acc, '%.5f')

    utils.print_log_acc_bwt(args, acc, lss)
    print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))


if __name__ == '__main__':
    main()
