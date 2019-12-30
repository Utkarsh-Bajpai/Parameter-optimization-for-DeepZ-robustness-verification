import os
import subprocess
import argparse
import verifier

MNIST=True
skip_nets=5



for i_net,net in enumerate(['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5']):
    if i_net<skip_nets:
        continue

    print("Evaluating network %s" % net)

    if MNIST:
        dir_path=os.path.join(os.path.dirname(__file__), '..', 'test_cases','MNIST_Dataset', net)
        specs = os.listdir(dir_path)
        #specs.sort()
        #filtered_spec = [specs[int(2 * i + 1)] for i in range(int(len(specs) / 2))]
        for i_spec,spec in enumerate(specs):
            if i_spec>10:
                break
            file_path=os.path.join(dir_path,spec)
            true_label, pixel_values, eps=verifier.read_spec(file_path)
            eps=eps
            verified,time_passed=verifier.core_analysis(net,true_label,pixel_values,eps,VERBOSE=1)
            if verified:
                print(spec)

    else:
        dir_path=os.path.join(os.path.dirname(__file__),'..','test_cases',net)
        specs=os.listdir(dir_path)
        for spec in specs:
            file_path=os.path.join(dir_path,spec)
            command = "python verifier.py --net %s --spec \"%s\"" % (net,file_path)
            process=subprocess.Popen(command,stdout=subprocess.PIPE)
            stdout = process.communicate()[0]
            print('STDOUT:{}'.format(stdout))

