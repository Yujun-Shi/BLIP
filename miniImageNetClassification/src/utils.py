import os
import numpy as np
import gzip
import pickle
from copy import deepcopy
from networks.quant_layer import Linear_Q, Conv2d_Q

########################################################################################################################
def print_arguments(args):
    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)

def make_directories(args):
    if args.output=='':
        args.output = '{}_{}'.format(args.experiment,args.approach)
        print (args.output)
    checkpoint = os.path.join(args.checkpoint_dir, args.output)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists(checkpoint):os.mkdir(checkpoint)
    print("Results will be saved in ", checkpoint)

    return checkpoint

def print_log_acc_bwt(args, acc, lss):

    print('*'*100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t',end=',')
        for j in range(acc.shape[1]):
            print('{:5.4f}% '.format(acc[i,j]),end=',')
        print()

    avg_acc = np.mean(acc[acc.shape[0]-1,:])
    print ('ACC: {:5.4f}%'.format(avg_acc))
    print()
    print()

    bwt = (acc[-1] - np.diag(acc)).mean()
    print ('BWT : {:5.2f}%'.format(bwt))

    print('*'*100)
    print('Done!')

    logs = {}
    # save results
    logs['name'] = args.experiment
    logs['taskcla'] = args.taskcla
    logs['acc'] = acc
    logs['loss'] = lss
    logs['bwt'] = bwt
    logs['rii'] = np.diag(acc)
    logs['rij'] = acc[-1]

    # pickle
    path = os.path.join(args.checkpoint, '{}_{}_seed_{}_F_prior_{}.p'.format(args.experiment, args.approach, args.seed, str(args.F_prior)))
    with open(path, 'wb') as output:
        pickle.dump(logs, output)

    print ("Log file saved in ", path)
    return avg_acc, bwt

# BLIP specific function
# check how many bits (capacity) has been used
def used_capacity(model, max_bit):
    used_bits = 0
    total_bits = 0
    for m in model.features.modules():
        if isinstance(m, Linear_Q) or isinstance(m, Conv2d_Q):
            used_bits += m.bit_alloc_w.sum().item()
            total_bits += max_bit*m.weight.numel()
            if m.bias is not None:
                used_bits += m.bit_alloc_b.sum().item()
                total_bits += max_bit*m.bias.numel()
    print('used capacity: ', float(used_bits)/float(total_bits))
