import argparse
import torch
from networks import FullyConnected, Conv, UnitTest
from transformerLayers import translate_nn
import time
import numpy as np
import torch.nn as nn
import os
import transformerLayers
import math



DEVICE = 'cpu'
INPUT_SIZE = 28
MAX_ITER=10000
MAX_TIME=120
VERBOSE=0
DEBUG=False
limit_optim_depth=False
LR_SCHEDULER=True
# TODO set DEBUG to False


# Verbosity:
# 0 - only verified / not verified
# 1 - Label where verification failed, if applicable, passed time and success of adverserial attack
# 2 - success statement whenever we verify against a label
# 3 - optimization progress
# 4 - backward pass analysis if in DEBUG mode


def analyze(net, inputs, eps, true_label,verbose=2):
    input_size=tuple(inputs.squeeze().shape)
    T_net=translate_nn(net,true_label,input_size)
    input_zono=make_zono(inputs,eps,clip=True)
    verified_flag=optimizePairwise(T_net, input_zono,true_label,verbose)
    return  verified_flag


def optimizePairwise(net, input_zono,true_label,verbose):
    start_t=time.time()

    if DEBUG:
        parameters_0 = []
        param_requires_grad=[]
        with torch.no_grad():
            for param in net.parameters():
                parameters_0.append(np.array(param.detach()))
                param_requires_grad.append(param.requires_grad)
        param_requires_grad=np.array(param_requires_grad)

    fwd_start=time.time()
    PairwiseDiff = net(input_zono)
    fwd_time=time.time()-fwd_start
    Verified=np.array(PairwiseDiff>0)##[False]*(n_labels-1)

    learning_rate = 2e-2
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=[0.90, 0.999])

    n_steps=int(np.ceil(MAX_TIME/(3.*fwd_time+1e-6)))
    if VERBOSE >2:
        print("maximum of %d steps predicted"%n_steps)
    n_steps=max([20, n_steps])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate * 3, total_steps=n_steps, pct_start=0.3,
                                                    base_momentum=0.85, max_momentum=0.90, div_factor=3.0,
                                                   final_div_factor=12.0,anneal_strategy='linear')

    conv_layers = [isinstance(x, transformerLayers.Conv2DTransformer) for x in list(net._modules.values())]
    conv_net = np.array(conv_layers).any()

    if conv_net and limit_optim_depth:
        relu_layers = [isinstance(x, transformerLayers.ReLuTransformer) for x in list(net._modules.values())]
        relu_conv_layers = [False] + [(conv_layers[i] and relu_layers[i + 1]) for i in range(len(conv_layers) - 1)]
        n_relu_conv=np.array(relu_conv_layers).sum()

        i_relu_conv=0
        with torch.no_grad():
            for relu_conv, param in zip(relu_conv_layers, net.parameters()):
                parameters_0.append(np.array(param.detach()))
                if relu_conv:
                    i_relu_conv+=1
                    if n_relu_conv-i_relu_conv>1:
                        param.requires_grad_(False)

    while (Verified==0).any():
        # get label with the largest but still negative pairwise difference
        label=[x for x in np.argsort(PairwiseDiff.detach()) if (int(x) in np.array(range(9))[Verified==0])][-1]
        t = 0

        if true_label <= label:
            print_label = label + 1
        else:
            print_label = label

        converged=False
        hist_loss=torch.zeros(size=(100,),dtype=torch.float32)

        while(PairwiseDiff[label]<0 and t<MAX_ITER and (time.time()-start_t)<MAX_TIME) and not converged:
            t+=1

            #loss=-PairwiseDiff[label]
            #loss=-PairwiseDiff[Verified==0].sum()
            loss=-PairwiseDiff[Verified==0].sum()-PairwiseDiff[label]*math.pow(t,1./2.)

            if DEBUG:
                with torch.autograd.profiler.profile() as prof:
                    loss.backward()
                if VERBOSE >3:
                    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            else:
                loss.backward()

            optimizer.step()

            with torch.no_grad():
                for param in net.parameters():
                    if param.requires_grad == True:
                        param.clamp_(0, 1)

            optimizer.zero_grad()
            PairwiseDiff = net(input_zono)

            if conv_net and LR_SCHEDULER:
                scheduler.step(np.min([t-1,n_steps-1]))
                if VERBOSE>2:
                    print("Learning rate in step %d is %.6f"%(t,optimizer.param_groups[0]['lr']))

            Verified=np.logical_or((PairwiseDiff > 0),Verified)

            if verbose > 2 :#and t % 10 == 9:
                print("Optimizing for label: %d \t Lower bound after %d steps: [%s]"
                      %(print_label,t, np.array2string(np.array(PairwiseDiff.detach()),precision=2)))
            if DEBUG:
                hist_loss[1:]=torch.clone(hist_loss[0:-1])
                hist_loss[0]=loss
                if t>100:
                    converged=((hist_loss.min()-hist_loss.max())/loss).abs()<1e-2

        if PairwiseDiff[label]>0:
            if DEBUG:
                parameters_1 = []
                with torch.no_grad():
                    for param in net.parameters():
                        parameters_1.append(np.array(param.detach()))

                var_unchanged=np.array([np.array(np.array(x1)==np.array(x2)).all() for (x1,x2) in zip(parameters_1,parameters_0)])
                if not var_unchanged[~param_requires_grad].all():
                    print("Variables changed, that shouldn't change")

            if verbose >1:
                print("Pairwise difference target achieved for label %d \t Lower bound: %f" %(print_label,PairwiseDiff[label]))
        else:
            if verbose >0:
                print("Pairwise difference verification failed at label %d after %d setps \t Lower bound: %f" %(print_label,t,PairwiseDiff[label]))

            return False

    if all(Verified):
        return True
    else:
        return False


def make_zono(inputs_r,eps,clip=True):
    inputs = torch.clone(inputs_r).squeeze()
    A_temp = torch.ones_like(inputs)
    A_temp = A_temp*eps
    if clip:
        dA = torch.max(inputs + A_temp - 1, torch.zeros_like(A_temp)) / 2
        A_temp -= dA
        inputs -= dA
        dA = torch.min(inputs - A_temp , torch.zeros_like(A_temp)) / 2
        A_temp += dA
        inputs -= dA

    A = torch.zeros((np.prod(inputs.shape),1) + tuple(inputs.shape), dtype=torch.float32) #eps space dim x Channels x space dim

    k = 0
    for i in range(inputs.shape[0]):
        if len(inputs.shape) > 1:
            for j in range(inputs.shape[1]):
                A[k,0,i, j] = A_temp[i,j]
                k += 1
        else:
            A[i,0,i] = A_temp[i]

    inputs=inputs.unsqueeze(0)
    return (inputs,A)


def fgsm_untargeted(model, x, label, eps):
    input=x.clone().detach_() #remove all dependencies
    input.requires_grad=True
    logits=model(input)
    model.zero_grad()
    target=torch.LongTensor([label])
    loss=nn.CrossEntropyLoss()(logits, target)
    loss.backward()
    grad = input.grad.reshape([28, 28])
    x=np.fmax(np.fmin((x+eps*np.sign(grad)).detach(),1),0)
    return x

def pgd_untargeted(model, x, label, k, eps, eps_step):
    x_1 = x
    for i in range(k):
        x_n = fgsm_untargeted(model, x, label, eps_step)
        x = x_1 + torch.clamp(x_n - x_1, -eps, +eps)
        x = torch.clamp(x, 0, 1)
    x = np.fmax(np.fmin(x, 1), 0)
    return x


def runPGD(net, inputs, eps, true_label):
    attack_successful=False
    k=200
    n=5
    eps_step=eps/10
    for i in range(n):
        input_adv=pgd_untargeted(net, inputs, true_label, k, eps, eps_step)
        out_adv = net(input_adv)
        pred_label_adv = out_adv.max(dim=1)[1].item()
        attack_successful=(pred_label_adv != true_label)
        if attack_successful:
            break
    return attack_successful


def runUnitTest():
    net = UnitTest(DEVICE).to(DEVICE)
    inputs = torch.FloatTensor(np.array([[0.1,.2],[0.4,0.5]])).view(1,1,2,2).to(DEVICE)
    eps=0.05
    true_label=0
    analyze(net, inputs, eps, true_label)


def core_analysis(net_str,true_label,pixel_values,eps,VERBOSE=0):
    start=time.time()

    if net_str == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 10]).to(DEVICE)
    elif net_str == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 50, 10]).to(DEVICE)
    elif net_str == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif net_str == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif net_str == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [400, 200, 100, 100, 10]).to(DEVICE)
    elif net_str == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif net_str == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif net_str == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [150, 10], 10).to(DEVICE)
    elif net_str == 'conv4':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    elif net_str == 'conv5':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)

    net.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),'../mnist_nets/%s.pt' % net_str), map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(np.array(pixel_values).reshape((1, 1, INPUT_SIZE, INPUT_SIZE))).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    verified=analyze(net, inputs, eps, true_label,verbose=VERBOSE)

    if verified:
        print('verified')
    else:
        print('not verified')

    end = time.time()

    if VERBOSE>0:
        print("time passed: %f s" % (end - start))

        #! TODO remove adverserial example search
        Adv_exmp_found=runPGD(net, inputs, eps, true_label)
        print("Adverserial example found : %s" %Adv_exmp_found)

    return verified,start-end

def read_spec(spec_path):
    with open(spec_path, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(round(float(lines[0]),0))
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(spec_path[:-4].split('/')[-1].split('_')[-1])
    return true_label,pixel_values,eps


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepZ relaxation')
    parser.add_argument('--net',
                        type=str,
                        choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
                        required=True,
                        help='Neural network to verify.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    true_label, pixel_values, eps=read_spec(args.spec)

    core_analysis(args.net,true_label,pixel_values,eps,VERBOSE=VERBOSE)

if __name__ == '__main__':
    # runUnitTest()
    main()
