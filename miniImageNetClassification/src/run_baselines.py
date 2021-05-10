import math
import sys,os,argparse,time
import numpy as np
import torch
import torch.nn as nn
import utils
from datetime import datetime

def main():
    tstart=time.time()

    parser=argparse.ArgumentParser(description='xxx')

    # Data parameters
    parser.add_argument('--seed',               default=0,               type=int,   help='(default=%(default)d)')
    parser.add_argument('--device',             default='cuda:0',        type=str,   help='gpu id')
    parser.add_argument('--approach',           default='lwf',           type=str,   help='approach used')
    parser.add_argument('--experiment',         default='MI',            type=str)
    parser.add_argument('--data_dir',           default='data',          type=str,   help='data directory')
    parser.add_argument('--ntasks',             default=10,              type=int)
    parser.add_argument('--pc-valid',           default=0.02,            type=float)
    parser.add_argument('--workers',            default=4,               type=int)

    # Training parameters
    parser.add_argument('--output',             default='',              type=str,   help='')
    parser.add_argument('--checkpoint_dir',     default='checkpoints/',  type=str,   help='')
    parser.add_argument('--nepochs',            default=200,             type=int,   help='')
    parser.add_argument('--sbatch',             default=64,              type=int,   help='')
    parser.add_argument('--lr',                 default=0.05,            type=float, help='')
    parser.add_argument('--momentum',           default=0.9,             type=float)
    parser.add_argument('--weight-decay',       default=0.0,             type=float)
    parser.add_argument('--resume',             default='no',            type=str,   help='resume?')
    parser.add_argument('--sti',                default=0,               type=int,   help='starting task?')
    parser.add_argument('--mul',                default=2,               type=int)

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
    if args.approach == 'ewc':
        from approaches import ewc as approach
    elif args.approach == 'sgd':
        from approaches import sgd as approach
    elif args.approach == 'sgd-frozen':
        from approaches import sgd_frozen as approach
    elif args.approach == 'imm-mode':
        from approaches import imm_mode as approach
    elif args.approach == 'lwf':
        from approaches import lwf as approach
    else:
        raise NotImplementedError("approach currently not implemented")

    # Args -- Network
    if args.approach != 'hat':
        from networks import alexnet as network
    else:
        from networks import alexnet_hat as network

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

        appr.save_model(task)
        # Test
        for u in range(task+1):
            data_u = datagen.dataloaders[u]
            test_loss, test_acc=appr.eval(u, data_u['test'])
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.3f}% <<<'.format(u,data_u['name'],test_loss,100*test_acc))
            acc[task,u]=test_acc
            lss[task,u]=test_loss

        # Save
        print('Save at '+args.checkpoint)
        np.savetxt(os.path.join(args.checkpoint,'{}_{}.txt'.format(args.approach,args.seed)), acc, '%.5f')

    utils.print_log_acc_bwt(args, acc, lss)
    print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))


if __name__ == '__main__':
    main()
