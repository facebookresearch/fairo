import argparse
import baselines
import torch, os
import torch.nn as nn


class Arguments(baselines.Arguments):
    @property
    def default(self):
        return {
            'input_data': './data/30hz.npz',
            'pretrained': './model/pretrained.npz',
            'features': 'VGGSoftmax',
            'BATCH_SIZE': 64,
            'EPOCHS': 100,
            'LR': 1e-3,
            'SAVE_FREQ': 20,
            'H': 1,
            'TRAJ': False,
            'LSTM': False,
        }


def train_bc(args):
    # create output folder
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # load dataset depending on arguments
    if args.TRAJ:
        train, test = baselines.datasets.traj_dataset(args.input_data, args.BATCH_SIZE)
    elif args.H == 1:
        train, test, _ = baselines.datasets.state_action_dataset(args.input_data, args.BATCH_SIZE)
    else:
        train, test = baselines.datasets.snippet_dataset(args.input_data, args.BATCH_SIZE, args.H)
  
    # build network and restore weights
    features = baselines.get_network(args.features)
    features.load_state_dict(torch.load(args.pretrained))
    Model = baselines.net.RNNPolicy if args.LSTM else baselines.net.CNNPolicy
    policy = Model(features, H=args.H).cuda()


    # build optim
    train_metric, test_metric = baselines.Metric(), baselines.Metric()
    optim = torch.optim.Adam(policy.parameters(), lr=args.LR)
    loss = nn.SmoothL1Loss()
    for e in range(args.EPOCHS):
        train_metric.reset(); test_metric.reset()
        policy = policy.train()
        for i, s, acs in train:
            optim.zero_grad()
            acs_hat = policy(i.cuda(), s.cuda())
            train_loss = loss(acs_hat, acs.cuda())
            train_loss.backward()
            optim.step()
            train_metric.add(train_loss.item())
            print('epoch {} \t train {:.6f} \t\t'.format(e, train_loss.item()), end='\r')
        
        policy = policy.eval()
        for i, s, acs in test:
            with torch.no_grad():
                acs_hat = policy(i.cuda(), s.cuda())
                test_metric.add(loss(acs_hat, acs.cuda()).item())
        print('epoch {} \t train {:.6f} \t test {:.6f}'.format(e, train_metric.mean, test_metric.mean))

        if (e + 1) % args.SAVE_FREQ == 0 or e + 1 == args.EPOCHS:
            torch.save(policy.state_dict(), args.output_folder + '/policy_epoch{}.pt'.format(e))


if __name__ == '__main__':
    args = baselines.args2cmd(Arguments)
    train_bc(args)
