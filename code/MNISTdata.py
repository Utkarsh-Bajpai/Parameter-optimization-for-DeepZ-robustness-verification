import torch
import torchvision
import torchvision.datasets as datasets
import numpy as np
from networks import FullyConnected, Conv
import torch.nn as nn
import os
import pandas as pd
import math

DEVICE = 'cpu'
INPUT_SIZE = 28

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
    k=50
    n=1
    eps_step=eps/5
    for i in range(n):
        input_adv=pgd_untargeted(net, inputs, true_label, k, eps, eps_step)
        out_adv = net(input_adv)
        pred_label_adv = out_adv.max(dim=1)[1].item()
        attack_successful=(pred_label_adv != true_label)
        if attack_successful:
            break
    return attack_successful

print(torch.__version__)
mnist_testset = datasets.MNIST(root='./data', train = False, download = True, transform = None)
mnist_trainset =  datasets.MNIST(root='./data', train = True, download = True, transform = None)

skip_nets=0
choices = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5']
eps_steps=33
eps_min=0.05
eps_max=0.21
eps_space=np.round(np.linspace(eps_min,eps_max,eps_steps),5)
#eps_space=np.round(np.logspace(math.log10(eps_min),math.log10(eps_max),eps_steps),4)

n_examples=100

if skip_nets>0:
    try:
        filename = os.path.join(os.path.dirname(__file__), '..', 'test_cases', 'MNIST_Dataset','Overview.txt')
        adv_ex_found=pd.read_csv(filename,index_col=0)
        assert adv_ex_found.columns==eps_space
    except:
        adv_ex_found = pd.DataFrame(np.zeros(shape=(len(choices), eps_steps)), columns=eps_space, index=choices)
        print("Created new overview")
else:
    adv_ex_found=pd.DataFrame(np.zeros(shape=(len(choices),eps_steps)),columns=eps_space,index=choices)

for i_net, net in enumerate(choices):
    if i_net<skip_nets:
        continue
    strnet = net
    if net == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 10]).to(DEVICE)
    elif net == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 50, 10]).to(DEVICE)
    elif net == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif net == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif net == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [400, 200, 100, 100, 10]).to(DEVICE)
    elif net == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif net == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif net == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [150, 10], 10).to(DEVICE)
    elif net == 'conv4':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    elif net == 'conv5':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)

    net.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '../mnist_nets/%s.pt' % strnet),
                                   map_location=torch.device(DEVICE)))
    dirname = os.path.join(os.path.dirname(__file__), '..', 'test_cases', 'MNIST_Dataset', strnet)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)


    for i, row in enumerate(mnist_testset):
        if np.sum(adv_ex_found.iloc[i_net, :])==n_examples:
            break
        #print(row)
        inputsnp = np.array(row[0])/255.
        inputs = torch.FloatTensor(inputsnp)
        #print(type(inputs))
        true_label = row[1]
        inputsnp = inputsnp.flatten()
        inputsnp = np.insert(inputsnp, 0, true_label, axis=0)

        outs = net(inputs)
        pred_label = outs.max(dim=1)[1].item()
        if (pred_label == true_label):
            eps_prev=eps_space[-1]
            for j in range(len(eps_space)):
                eps=eps_space[-j-1]
                Adv_exmp_found = runPGD(net, inputs, eps, true_label)
                if np.array(Adv_exmp_found)==0:
                    if j>0:
                        filename = os.path.join(os.path.dirname(__file__),'..','test_cases','MNIST_Dataset',strnet,'img%d_%.5f.txt'%(i,eps_prev))
                        with open(filename, 'wb') as f:
                            np.savetxt(f, inputsnp, fmt='%.15f')
                        print("Adverserial example found for net %s and image %d at eps: %.5f" % (strnet,i,eps_prev))
                        adv_ex_found.iloc[i_net,-j]+=1
                    break
                eps_prev=eps
    filename = os.path.join(os.path.dirname(__file__), '..', 'test_cases', 'MNIST_Dataset','Overview.txt')
    adv_ex_found.to_csv(filename)