import argparse
import baselines
import torch, os
import torch.nn as nn
import matplotlib.pyplot as plt


class Arguments(baselines.Arguments):
    @property
    def default(self):
        return {
            'input_data': './data/30hz.npz',
            'pretrained': './model/pretrained.npz',
            'features': 'VGGSoftmax',
            'BATCH_SIZE': 64,
            'EPOCHS': 5000,
            'LR': 1e-3,
            'SAVE_FREQ': 1000,
            'N': 270,
            'H': 299
        }


def train_ndp(args):
    # create output folder
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # load dataset
    train, test = baselines.datasets.traj_dataset(args.input_data, args.BATCH_SIZE)


    # build network and restore weights
    features = baselines.get_network(args.features)
    features.load_state_dict(torch.load(args.pretrained))
    policy = baselines.net.DMPNet(features, args.N, args.H).cuda()


    # build optim
    train_metric, test_metric = baselines.Metric(), baselines.Metric()
    optim = torch.optim.Adam(policy.parameters(), lr=args.LR)
    loss = nn.MSELoss()
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
            print('epoch {} \t train {:.6f} \t\t'.format(e, train_metric.mean), end='\r')
        
        policy = policy.eval()
        for i, s, acs in test:
            with torch.no_grad():
                acs_hat = policy(i.cuda(), s.cuda())
                test_metric.add(loss(acs_hat, acs.cuda()).item())
            
        print('epoch {} \t train {:.6f} \t test {:.6f}'.format(e, train_metric.mean, test_metric.mean))

        if (e + 1) % args.SAVE_FREQ == 0 or e + 1 == args.EPOCHS:
            acs_hat = acs_hat.cpu().numpy()
            acs = acs.cpu().numpy()
            plt.close()
            fig, axs = plt.subplots(min(4, acs.shape[0]), 7, figsize=(30, 15))
            for k in range(axs.shape[0]):
                for i in range(7):
                    axs[k, i].plot(acs_hat[k, :, i], label='global pol test')
                    axs[k, i].plot(acs[k, :, i], label='test')
            plt.legend()
            plt.savefig(args.output_folder + '/plots_{}.png'.format(e))
            torch.save(policy.state_dict(), args.output_folder + '/policy_epoch{}.pt'.format(e))


if __name__ == '__main__':
    args = baselines.args2cmd(Arguments)
    train_ndp(args)
