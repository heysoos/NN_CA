import argparse
from model import *
from pytorch_memlab import profile, profile_every, MemReporter

import matplotlib.pyplot as plt

from tqdm import tqdm
import time
import copy
import sys

SEED = 10000

## DEFAULT VALUES
CHANNEL_N = 16
RADIUS = 1
NUM_FILTERS = 10
HIDDEN_N = 128

EMBED_KERNEL = 5

EPOCHS = 10
NUM_POP = 30
RES = 50

# reporter = MemReporter()

def create_settings():
    args = parse()
    # --- CONSTANTS ----------------------------------------------------------------+
    settings = {}

    # MODEL SETTINGS
    settings['CHANNEL_N'] = args.channel_n
    settings['RADIUS'] = args.radius
    settings['NUM_FILTERS'] = args.num_filters
    settings['HIDDEN_N'] = args.hidden_n

    settings['EMBED_KERNEL'] = args.embed_kernel

    settings['BATCH_SIZE'] = 1

    # TRAINING SETTINGS
    settings['EPOCHS'] = args.epochs
    settings['NUM_POP'] = args.pop

    # SIMULATION SETTINGS
    settings['save'] = args.save_data
    settings['plot'] = args.plot
    settings['RES'] = args.res
    settings['SEED'] = SEED

    return settings

def parse():
    parser = argparse.ArgumentParser(description=
                                     'Open-ended evolution in neural cellular automata'
                                     '------Default values-----'
                                     f'save_data=True, plot=True, channel_n={CHANNEL_N}, radius={RADIUS}, '
                                     f'num_filters={NUM_FILTERS}, hidden_n={HIDDEN_N}, embed_kernel={EMBED_KERNEL}, '
                                     f'epochs={EPOCHS}, pop={NUM_POP}'
                                     )
    parser.add_argument('-p', '--pop', dest='pop', type=int,
                        help='Number of CAs')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int,
                        help='Number of training epochs')

    parser.add_argument('-c', '--channel_n', dest='channel_n', type=int,
                        help='Number of CA channels')
    parser.add_argument('-r', '--radius', dest='radius', type=int,
                        help='Radius of CA kernels.')
    parser.add_argument('-nf', '--num_filters', dest='num_filters', type=int,
                        help='Number of kernels in CA network.')
    parser.add_argument('-hn', '--hidden_n', dest='hidden_n', type=int,
                        help='Number of neurons in hidden layer')
    parser.add_argument('-ek', '--embed_kernel', dest='embed_kernel', type=int,
                        help='Size of embedder kernel.')

    parser.add_argument('-s', '--save', dest='save_data', action='store_true', help="Checkpoint models during training.")
    parser.add_argument('-pl', '--plot', dest='plot', action='store_true',
                        help="Make figures")
    parser.add_argument('-res', dest='res', type=int,
                        help='Grid resolution.')

    parser.add_argument('-n', '--name', dest='savename', help='Optional name for the folder')

    #-n does not do anything in the code as input arguments already define name of folder. Practical nonetheless.

    parser.set_defaults(pop=NUM_POP, epochs=EPOCHS,
                        channel_n=CHANNEL_N, radius=RADIUS, num_filters=NUM_FILTERS,
                        hidden_n=HIDDEN_N, embed_kernel=EMBED_KERNEL,
                        save_data=True, plot=True, res=RES)

    args = parser.parse_args()
    return args


# --- MAIN ---------------------------------------------------------------------+
# @profile_every(1)
def train(settings):

    ####### INITIALIZE ########
    seed = settings['SEED']
    np.random.seed(seed)
    torch.manual_seed(seed)

    print('Initializing models...')

    CA_args = (settings['CHANNEL_N'], settings['RADIUS'], settings['NUM_FILTERS'], settings['HIDDEN_N'])
    Embed_args = (settings['EMBED_KERNEL'],)
    CA_orig = CAModel(*CA_args)

    CAs = [CA_orig]
    for i in range(settings['NUM_POP'] - 1):
        CAs.append(copy.deepcopy(CA_orig))

    embed = Embedder(*Embed_args)
    if torch.cuda.is_available():
        for CA in CAs:
            CA.cuda()

        embed.cuda()

    tloss = nn.TripletMarginLoss()
    #############################
    optim_emb = torch.optim.Adam([p for p in embed.parameters()], lr=1e-3)
    Optims_CAs = [torch.optim.Adam([p for p in CA.parameters()], lr=1e-3) for CA in CAs]

    res = settings['RES']
    epochs = settings['EPOCHS']
    step_size = 1
    fire_rate = 1

    emb_loss = []
    total_dists = []
    mean_grads = []
    hard_negative_sum = []
    hard_frac = []

    index_list = np.arange(len(CAs))

    if settings['save']:
        s = sys.argv[1:]
        command_input = '_'.join([str(elem) for elem in s])
        folder = f'models/model_' + time.strftime("%Y%m%d-%H%M%S") + command_input
        CAfig_folder = path.join(folder, 'figs', 'CAs')
        statfig_folder = path.join(folder, 'figs', 'stats')

        makedirs(folder)
        makedirs(CAfig_folder)
        makedirs(statfig_folder)

        PATH = path.join(folder, 'epoch_' + str(0).zfill(4) + '.tar')

        copy_model(folder)
        torch.save({
            'epoch': 0,
            'model_state_dict': [CA.state_dict() for CA in CAs],
            'optimizer_state_dict': [optim.state_dict() for optim in Optims_CAs],
            'embedder_state_dict': embed.state_dict(),
            'embedder_optimizer_state_dict': optim_emb.state_dict(),
            'loss': None,
            'stats': None
        }, PATH)

    for epoch in range(epochs):
        total_loss = 0

        zs_1 = []
        zs_2 = []

        if epoch > 1:

            if settings['plot'] and settings['save']:
                fig, ax = plt.subplots(2, 3, figsize=(12, 6))
                ax[0, 0].plot(emb_loss, '.-', label='Tloss')
                ax[0, 1].plot(mean_grads, '.-', label='AvgGrad')
                ax[0, 2].plot(np.log10(total_dists), '.-', label='Sum of distances')
                ax[1, 0].plot(hard_negative_sum, '.-', label='Hard negtive mining sum')
                ax[1, 2].plot(hard_frac, '.-', label='Hard negative fraction')

                [a.legend() for i, a in enumerate(ax.flatten()) if i != 4]

                fig.suptitle(f'Epoch: {epoch}')
                STATPLOT_PATH = path.join(statfig_folder, '_epoch_' + str(epoch).zfill(4) + '_stats.png')
                plt.savefig(STATPLOT_PATH)
                plt.close()

            if epoch % 10 == 0:
                stats = {
                    'hneg_sum': hard_negative_sum,
                    'hfrac': hard_frac,
                    'mean_grad': mean_grads,
                    'total_dists': total_dists
                }

                if settings['save']:
                    PATH = path.join(folder, 'epoch_' + str(epoch).zfill(4) + '.tar')

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': [CA.state_dict() for CA in CAs],
                        'optimizer_state_dict': [optim.state_dict() for optim in Optims_CAs],
                        'embedder_state_dict': embed.state_dict(),
                        'embedder_optimizer_state_dict': optim_emb.state_dict(),
                        'loss': emb_loss,
                        'stats': stats
                    }, PATH)

        for CA_i in tqdm(range(len(CAs))):
            # pick most similar or random other CA
            if epoch == 0:
                # pick a random other CA
                # CA_B_i = np.random.permutation(index_list[index_list != CA_i])[0]
                CA_B_i = np.random.permutation(index_list)[0]
                CA_B = CAs[CA_B_i]
                CA_A = CAs[CA_i]
            else:
                # find the top-k most similar CA and compare vs. that

                CA_i_zstats = [item for item in tloss_prev if item[0] == CA_i]
                B_idxs = [item[1] for item in CA_i_zstats]
                k = len(CA_i_zstats)  # number of other CAs to compare to (change beta if you increase/decrease k)
                z_dists, nearest = torch.topk(torch.stack([item[2] for item in CA_i_zstats]),
                                              k=k)  # get the CAs that are most similar to CA_i
                beta = 2
                p = F.softmax(beta * z_dists, dim=0).cpu().detach().numpy()
                CA_B_i = B_idxs[np.random.choice(nearest.cpu().detach().numpy(), p=p)]

                # check for dead/frozen CAs, re-initialize if so (checks last 3 embeddings for differences)
                lookback_time = 3
                check_dead_A = (np.diff(zs_prev[CA_i].reshape(-1, 8).cpu().detach().numpy(), axis=0)).sum(axis=1)[
                               -lookback_time:].sum() == 0

                if check_dead_A:
                    CAs[CA_i] = CAModel(*CA_args).cuda()
                    Optims_CAs[CA_i] = torch.optim.Adam([p for p in CAs[CA_i].parameters()], lr=1e-3)
                    print(f'CA {CA_i} is dead. Reinitializing.')

                CA_A = CAs[CA_i]
                CA_B = CAs[CA_B_i]

            reporter = MemReporter(CAs[0])

            # reset IC
            x_A1 = torch.cuda.FloatTensor(np.random.standard_normal(size=(settings['CHANNEL_N'], res, res))).unsqueeze(0)
            x_A2 = torch.cuda.FloatTensor(np.random.standard_normal(size=(settings['CHANNEL_N'], res, res))).unsqueeze(0)
            x_B = torch.cuda.FloatTensor(np.random.standard_normal(size=(settings['CHANNEL_N'], res, res))).unsqueeze(0)

            for jj in range(1):

                optim_emb.zero_grad()
                Optims_CAs[CA_i].zero_grad()
                Optims_CAs[CA_B_i].zero_grad()

                # forward, save in memory
                z_A1 = []
                z_A2 = []
                z_B = []

                for ii in range(1, 51):
                    # check gradients
                    x_A1 = torch.tanh(CA_A.forward(x_A1, step_size=step_size, fire_rate=fire_rate))
                    x_A2 = torch.tanh(CA_A.forward(x_A2, step_size=step_size, fire_rate=fire_rate))
                    x_B = torch.tanh(CA_B.forward(x_B, step_size=step_size, fire_rate=fire_rate))

                    #                 x_A1 = CA_A.forward(x_A1, step_size=step_size, fire_rate=fire_rate)
                    #                 x_A2 = CA_A.forward(x_A2, step_size=step_size, fire_rate=fire_rate)
                    #                 x_B = CA_B.forward(x_B, step_size=step_size, fire_rate=fire_rate)

                    if ii % 5 == 0:
                        # embed, calculate loss
                        z_A1.append(embed.forward(x_A1[:, 0:4, :, :]))
                        z_A2.append(embed.forward(x_A2[:, 0:4, :, :]))
                        z_B.append(embed.forward(x_B[:, 0:4, :, :]))

                z_A1 = torch.cat(z_A1, 0)
                z_A2 = torch.cat(z_A2, 0)
                z_B = torch.cat(z_B, 0)

                loss = tloss(z_A1, z_A2, z_B)
                total_loss += loss.item() / len(CAs)
                loss.backward()

                # normalize gradients
                gradmax = 0
                for CA in [CA_A, CA_B]:
                    for p in CA.parameters():
                        if p.grad is not None:
                            gradnorm = torch.norm(p.grad)

                            if torch.abs(p.grad).max() > gradmax:
                                gradmax = torch.abs(p.grad).max()

                            p.grad = p.grad / (1e-8 + gradnorm)

                optim_emb.step()
                Optims_CAs[CA_i].step()
                Optims_CAs[CA_B_i].step()

                x_A1 = x_A1.detach()
                x_A2 = x_A2.detach()
                x_B = x_B.detach()

            # save embedding time-series for each CA (passes to zs_prev)
            zs_1.append(z_A1.detach().cpu())
            zs_2.append(z_A2.detach().cpu())

            ##### PLOTTING THINGS #####
            if settings['plot'] and settings['save'] and epoch % 3 == 0:

                nx_A1 = x_A1.cpu().detach().numpy()[0, 0:4, :, :].transpose(1, 2, 0)
                nx_A1 = rgba2rgb(nx_A1)
                nx_A1 = np.uint8(nx_A1 * 255.0)

                nx_A2 = x_A2.cpu().detach().numpy()[0, 0:4, :, :].transpose(1, 2, 0)
                nx_A2 = rgba2rgb(nx_A2)
                nx_A2 = np.uint8(nx_A2 * 255.0)

                nx_B = x_B.cpu().detach().numpy()[0, 0:4, :, :].transpose(1, 2, 0)
                nx_B = rgba2rgb(nx_B)
                nx_B = np.uint8(nx_B * 255.0)

                fig, axes = plt.subplots(1, 3, figsize=(12, 6))

                axes[0].imshow(nx_A1)
                axes[0].set_title(f'{CA_i}')
                axes[1].imshow(nx_A2)
                axes[2].imshow(nx_B)
                axes[2].set_title(f'{CA_B_i}')

                for ax in axes:
                    ax.set_axis_off()

                # fig.suptitle(f'{CA_i} Max. gradient: {gradmax:e}, Tloss: {loss:e}')

                CAIM_subfolder = path.join(CAfig_folder, 'epoch_' + str(epoch).zfill(4))
                if not path.exists(CAIM_subfolder):
                    makedirs(CAIM_subfolder)
                CAIM_PATH = path.join(CAIM_subfolder, 'epoch_' + str(epoch).zfill(4) + '_CA_' + str(CA_i).zfill(3) + '.png')
                plt.savefig(CAIM_PATH)
                plt.close()
                ###########################
            reporter.report()
            # print(torch.cuda.memory_stats(torch.cuda.current_device()))

        meangrad = 0
        for CA in CAs:
            for p in CA.parameters():
                if p.grad is not None:
                    meangrad += torch.abs(p.grad).mean().detach()
        mean_grads.append(meangrad.mean())

        emb_loss.append(total_loss)

        # used for hard-negative mining
        tloss_prev = []
        for i in range(len(zs_1)):
            for j in range(len(zs_1)):
                if j != i:
                    tloss_prev.append((i, j, tloss(zs_1[i], zs_2[i], zs_1[j]).detach().cpu() ))

        # stats
        zs_prev = torch.stack(zs_1).view(settings['NUM_POP'], -1).detach().cpu()
        dists = torch.cdist(zs_prev, zs_prev) / len(CAs)
        total_dists.append(dists.triu().sum().numpy())

        hard_negative = [tl[2].detach().cpu().numpy() for tl in tloss_prev if tl[2] > 0.]
        hard_negative_sum.append(np.stack(hard_negative).sum() / settings['NUM_POP'])
        hard_frac.append(len(hard_negative) / len(tloss_prev))

        print(f'{epoch}: tloss={emb_loss[-1]:.4f}, hardfrac={hard_frac[-1]:.4f}, distsum={total_dists[-1]:.2f}')

    pass


# --- RUN ----------------------------------------------------------------------+
if __name__ == '__main__':
    settings = create_settings()
    train(settings)

# --- END ----------------------------------------------------------------------+
