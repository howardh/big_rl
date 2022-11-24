import argparse
import os
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec
from matplotlib.animation import FuncAnimation, ArtistAnimation, PillowWriter, FFMpegWriter
from matplotlib.axes import Axes


def render_trajectory_video(
        data, rewards, steps, trail_length=5, 
        video_format='webm', fps=30,
        directory='./',
        x_lim=None, y_lim=None):
    plt.figure()
    ax = plt.gca()
    assert isinstance(ax, Axes)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    artists = []

    min_start_index = 0
    for end_index in tqdm(range(data.shape[0]), desc='Rendering trajectory video'):
        artists.append([])
        if steps[end_index] == 0:
            min_start_index = end_index
            continue
        for start_index in range(max(end_index-trail_length-1,min_start_index), end_index-1):
            x = data[start_index:start_index+2,0]
            y = data[start_index:start_index+2,1]
            alpha = 1 - (end_index - start_index - 1) / trail_length
            #print(x,y,alpha)
            a = ax.plot(x, y, '-',
                    alpha=alpha,
                    color='black'
            )
            artists[-1].append(a[0])

            r = rewards[start_index]
            if r > 0:
                artists[-1].append(ax.scatter(x[0], y[0], color='green', alpha=alpha))
            elif r < 0:
                artists[-1].append(ax.scatter(x[0], y[0], color='red', marker='X', alpha=alpha))

    print('Rendering video')
    filename = os.path.join(
            directory, 'trajectory_video.{}'.format(video_format))
    animation = ArtistAnimation(plt.gcf(), artists, interval=50, blit=True)
    if video_format == 'gif':
        writer = PillowWriter(fps=fps)
    elif video_format == 'mp4':
        writer = FFMpegWriter(fps=fps)
    elif video_format == 'webm':
        writer = FFMpegWriter(fps=fps, codec='libvpx-vp9')
    else:
        raise ValueError('Unknown video format: {}'.format(video_format))
    pbar = tqdm(total=len(artists), desc='Writing video')
    animation.save(filename, writer=writer, progress_callback=lambda *_: pbar.update())
    print(f'Saved video to {os.path.abspath(filename)}')
    plt.close()


def plot_trajectory_around_rewards(
        data, rewards, steps, trail_length=1, 
        directory='./',
        x_lim=None, y_lim=None):
    ##################################################
    # Note: The hidden states at index i is the hidden state that is produced by the model just before observing the reward at index i.

    # Plot trajectory just before reaching a positive reward
    plt.figure()
    ax = plt.gca()
    assert isinstance(ax, Axes)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_title('Trajectory before reaching a positive reward')
    for i,r in enumerate(rewards):
        if r <= 0:
            continue
        ax.scatter(data[i,0], data[i,1], color='green')
        for j in reversed(range(max(i-trail_length-1, 0), i)):
            if steps[j] == 0:
                continue
            x = data[j:j+2,0]
            y = data[j:j+2,1]
            alpha = 1 - (i - j - 1) / trail_length
            ax.plot(x, y, '-',
                    alpha=alpha,
                    color='black'
            )
    filename = os.path.join(directory, 'trajectory-before-positive.png')
    plt.savefig('trajectory-before-positive.png')
    print(f'Saved figure to {os.path.abspath(filename)}')
    plt.close()

    # Plot trajectory just before reaching a negative reward
    plt.figure()
    ax = plt.gca()
    assert isinstance(ax, Axes)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_title('Trajectory before reaching a negative reward')
    for i,r in enumerate(rewards):
        if r >= 0:
            continue
        ax.scatter(data[i,0], data[i,1], marker='X', color='red')
        for j in reversed(range(max(i-trail_length-1, 0), i)):
            if steps[j] == 0:
                continue
            x = data[j:j+2,0]
            y = data[j:j+2,1]
            alpha = 1 - (i - j - 1) / trail_length
            ax.plot(x, y, '-',
                    alpha=alpha,
                    color='black'
            )
    filename = os.path.join(directory, 'trajectory-before-negative.png')
    plt.savefig(filename)
    print(f'Saved figure to {os.path.abspath(filename)}')
    plt.close()

    # Plot trajectory just after reaching a positive reward
    plt.figure()
    ax = plt.gca()
    assert isinstance(ax, Axes)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_title('Trajectory after reaching a positive reward')
    for i,r in enumerate(rewards):
        if r <= 0:
            continue
        ax.scatter(data[i,0], data[i,1], color='green')
        for j in range(i, min(i+trail_length, data.shape[0])):
            if steps[j] == 0:
                continue
            x = data[j:j+2,0]
            y = data[j:j+2,1]
            alpha = 1 - (j - i) / trail_length
            ax.plot(x, y, '-',
                    alpha=alpha,
                    color='black'
            )
    filename = os.path.join(directory, 'trajectory-after-positive.png')
    plt.savefig(filename)
    print(f'Saved figure to {os.path.abspath(filename)}')
    plt.close()

    # Plot trajectory just after reaching a negative reward
    plt.figure()
    ax = plt.gca()
    assert isinstance(ax, Axes)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_title('Trajectory after reaching a negative reward')
    for i,r in enumerate(rewards):
        if r >= 0:
            continue
        ax.scatter(data[i,0], data[i,1], marker='X', color='red')
        for j in range(i, min(i+trail_length, data.shape[0])):
            if steps[j] == 0:
                continue
            x = data[j:j+2,0]
            y = data[j:j+2,1]
            alpha = 1 - (j - i) / trail_length
            print(x,y,alpha,j,i)
            ax.plot(x, y, '-',
                    alpha=alpha,
                    color='black'
            )
    filename = os.path.join(directory, 'trajectory-after-negative.png')
    plt.savefig(filename)
    print(f'Saved figure to {os.path.abspath(filename)}')
    plt.close()


def plot_reward_triggered_average(data, rewards, steps, directory='./'):
    window_size = 51 # Must be odd

    rta_data = []
    for i,r in enumerate(rewards):
        if r <= 0:
            continue
        if i < window_size // 2:
            continue
        if i > data.shape[0] - window_size // 2:
            continue
        if steps[i + window_size // 2] < window_size // 2:
            continue
        rta_data.append(data[i-window_size//2:i+window_size//2+1,:])
    rta_data = np.stack(rta_data, axis=0).mean(axis=0)
    n = rta_data.shape[1]
    colours = ['#ff6188', '#a9dc76', '#78dce8'] # Monokai's red/green/blue

    plt.figure()
    gs = (matplotlib.gridspec.GridSpec(n,1))
    ax = []
    for i in range(n):
        ax.append(plt.subplot(gs[i]))
        # Plot data
        ax[-1].plot(range(-window_size//2, window_size//2), rta_data[:,i],
                color=colours[i%len(colours)])
        # Remove the y-axis ticks and tick labels
        ax[-1].set_yticks([])
        ax[-1].set_yticklabels([])
        # For all but the last subplot
        if i != rta_data.shape[1] - 1:
            # Remove the x-axis tick labels
            ax[-1].set_xticklabels([])
            # Remove spines
            ax[-1].spines['right'].set_visible(False)
            ax[-1].spines['top'].set_visible(False)
            ax[-1].spines['left'].set_visible(False)
            ax[-1].spines['bottom'].set_visible(False)
        else:
            # Remove spines
            ax[-1].spines['right'].set_visible(False)
            ax[-1].spines['top'].set_visible(False)
            ax[-1].spines['left'].set_visible(False)
        # Make background transparent
        ax[-1].patch.set_alpha(0.0)
        # Set vertical gridlines
        ax[-1].xaxis.grid(True)
    ax[0].set_title('Reward-triggered averaging (positive reward)')
    ax[-1].set_xlabel('Time relative to reward')
    gs.update(hspace = -0.5)
    plt.tight_layout()
    filename = os.path.join(directory, 'rta-positive.png')
    plt.savefig(filename)
    print(f'Saved figure to {os.path.abspath(filename)}')

    # Reward-triggered averaging (negative reward)
    rta_data = []
    for i,r in enumerate(rewards):
        if r >= 0:
            continue
        if i < window_size // 2:
            continue
        if i > data.shape[0] - window_size // 2:
            continue
        rta_data.append(data[i-window_size//2:i+window_size//2+1,:])
    rta_data = np.stack(rta_data, axis=0).mean(axis=0)
    n = rta_data.shape[1]
    colours = ['#ff6188', '#a9dc76', '#78dce8'] # Monokai's red/green/blue

    plt.figure()
    gs = (matplotlib.gridspec.GridSpec(n,1))
    ax = []
    for i in range(n):
        ax.append(plt.subplot(gs[i]))
        # Plot data
        ax[-1].plot(range(-window_size//2, window_size//2), rta_data[:,i],
                color=colours[i%len(colours)])
        # Remove the y-axis ticks and tick labels
        ax[-1].set_yticks([])
        ax[-1].set_yticklabels([])
        # For all but the last subplot
        if i != rta_data.shape[1] - 1:
            # Remove the x-axis tick labels
            ax[-1].set_xticklabels([])
            # Remove spines
            ax[-1].spines['right'].set_visible(False)
            ax[-1].spines['top'].set_visible(False)
            ax[-1].spines['left'].set_visible(False)
            ax[-1].spines['bottom'].set_visible(False)
        else:
            # Remove spines
            ax[-1].spines['right'].set_visible(False)
            ax[-1].spines['top'].set_visible(False)
            ax[-1].spines['left'].set_visible(False)
        # Make background transparent
        ax[-1].patch.set_alpha(0.0)
        # Set vertical gridlines
        ax[-1].xaxis.grid(True)
    ax[0].set_title('Reward-triggered averaging (negative reward)')
    ax[-1].set_xlabel('Time relative to reward')
    gs.update(hspace = -0.5)
    plt.tight_layout()
    filename = os.path.join(directory, 'rta-negative.png')
    plt.savefig(filename)
    print(f'Saved figure to {os.path.abspath(filename)}')


def plot_data_2d(data, steps, directory='./', filename='plot.png'):
    fig = plt.figure()
    p = plt.scatter(data[:,0], data[:,1],
        c=steps # Colour by time
    )
    cbar = fig.colorbar(p)
    cbar.ax.set_ylabel('Steps')
    filename = os.path.join(directory, filename)
    plt.savefig(filename)
    plt.close()
    print(f'Saved scatter plot to {os.path.abspath(filename)}')


def plot_data_3d(data, steps, directory='./', filename='plot.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    assert isinstance(ax, Axes)
    p = ax.scatter(data[:,0], data[:,1], data[:,2],
        c=steps # Colour by time
    )
    cbar = fig.colorbar(p)
    cbar.ax.set_ylabel('Steps', rotation=270)
    filename = os.path.join(directory, filename)
    plt.savefig(filename)
    plt.close()
    print(f'Saved 3D scatter plot to {os.path.abspath(filename)}')


def plot_data_3d_video_rotation(data, steps, directory='./', filename='video.webm', fps=30, num_frames=360):
    """ Produce a video of a 3D scatter plot rotating around the vertical axis. """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    assert isinstance(ax, Axes)
    p = ax.scatter(data[:,0], data[:,1], data[:,2],
        c=steps # Colour by time
    )
    cbar = fig.colorbar(p)
    cbar.ax.set_ylabel('Steps', rotation=270)

    def animate(i):
        ax.view_init(elev=ax.elev, azim=i)
        return fig,

    video_format = filename.split('.')[-1]
    filename = os.path.join(directory, filename)
    animation = FuncAnimation(fig, animate, frames=num_frames, blit=True)
    if video_format == 'gif':
        writer = PillowWriter(fps=fps)
    elif video_format == 'mp4':
        writer = FFMpegWriter(fps=fps)
    elif video_format == 'webm':
        writer = FFMpegWriter(fps=fps, codec='libvpx-vp9')
    else:
        raise ValueError(f'Unknown video format {video_format}')
    pbar = tqdm(total=num_frames, desc='Writing video')
    animation.save(filename, writer=writer, progress_callback=lambda *_: pbar.update())

    print(f'Saved 3D scatter plot video to {os.path.abspath(filename)}')


def plot_data_3d_video_time(data, steps, directory='./', filename='video.webm', fps=30):
    """ Produce a video of a 3D scatter plot varying the opacity of the points based on their time step. """
    warnings.warn('This function is not working correctly. The points seem to be becoming opaque in the wrong order.')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    assert isinstance(ax, Axes)
    artists = []

    for t in tqdm(range(steps.max()), desc='Rendering trajectory video'):
        artists.append([])

        #s = 0.1 # Window size
        max_alpha = 1.0
        #alpha = np.exp(-((steps-t)/(steps.max()*s))**2)
        #alpha = alpha / alpha.max() * max_alpha
        alpha = np.abs(steps-t) < 1
        tqdm.write(f'Avg step: {steps[alpha]}')
        alpha = alpha * max_alpha

        a = ax.scatter(data[:,0], data[:,1], data[:,2],
                alpha=alpha,
                c=steps # Colour by time
        )
        #breakpoint()
        artists[-1].append(a)

        #tqdm.write(f'Alpha: {alpha[0]}')

    cbar = fig.colorbar(a) # type: ignore
    cbar.ax.set_ylabel('Steps', rotation=270)

    video_format = filename.split('.')[-1]
    filename = os.path.join(directory, filename)
    animation = ArtistAnimation(plt.gcf(), artists, interval=50, blit=True)
    if video_format == 'gif':
        writer = PillowWriter(fps=fps)
    elif video_format == 'mp4':
        writer = FFMpegWriter(fps=fps)
    elif video_format == 'webm':
        writer = FFMpegWriter(fps=fps, codec='libvpx-vp9')
    else:
        raise ValueError('Unknown video format: {}'.format(video_format))
    pbar = tqdm(total=len(artists), desc='Writing video')
    animation.save(filename, writer=writer, progress_callback=lambda *_: pbar.update())
    print(f'Saved video to {os.path.abspath(filename)}')
    plt.close()


def plot_trajectory(data,
                    rewards,
                    tsne_seed=0,
                    #trail_length: int = 5,
                    video_format=None,
                    video_fps=30,
                    directory='./'):
    # TODO: Handle the episode boundaries.
    steps = np.concatenate([np.arange(len(r)) for r in rewards])
    data = np.concatenate(data, axis=0)
    rewards = np.concatenate(rewards, axis=0)

    # Dimensionality reduction with PCA
    pca = PCA(n_components=20) # Do 20 components so we can plot the variance explained
    pca.fit(data)

    explained_variance_ratio = pca.explained_variance_ratio_
    plt.figure()
    plt.plot(np.arange(1, 21), np.cumsum(explained_variance_ratio))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.grid()
    filename = os.path.join(directory, 'pca.png')
    plt.savefig(filename)
    plt.close()
    print(f'Saved PCA plot to {os.path.abspath(filename)}')

    pca = PCA(n_components=0.99)
    pca.fit(data) # XXX: Can I do this without calling `fit` again?
    #pca.n_components = 0.99
    data_pca = pca.transform(data)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=tsne_seed)
    data_tsne = tsne.fit_transform(data_pca)

    # Plot all points
    plot_data_2d(data_pca, steps, directory=directory,
            filename='trajectory-pca.png')
    plot_data_3d(data_pca, steps, directory=directory,
            filename='trajectory-pca-3d.png')
    plot_data_3d_video_rotation(data_pca, steps, directory=directory,
            filename='trajectory-pca-3d-video-rotation.webm')
    #plot_data_3d_video_time(data_pca, rewards, steps, directory=directory,
    #        filename='trajectory-pca-3d-video-time.webm')

    print(f'Saved scatter plot of PCA data to {os.path.abspath(filename)}')

    fig = plt.figure()
    p = plt.scatter(data_tsne[:,0], data_tsne[:,1], c=steps)
    cbar = fig.colorbar(p)
    cbar.ax.set_ylabel('Steps', rotation=270)
    filename = os.path.join(directory, 'trajectory-tsne.png')
    plt.savefig(filename)
    print(f'Saved scatter plot of t-SNE data to {os.path.abspath(filename)}')

    ax = plt.gca()
    assert isinstance(ax, Axes)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    plt.close()

    # Plot trajectory over the entire episode
    if video_format is not None:
        render_trajectory_video(
            data_tsne, rewards, steps,
            trail_length=5,
            video_format=video_format, fps=video_fps,
            directory=directory,
            x_lim=x_lim, y_lim=y_lim
        )

    # Plot trajectory around rewards
    plot_trajectory_around_rewards(
            data_tsne, rewards, steps,
            directory=directory,
            x_lim=x_lim, y_lim=y_lim
    )

    # Reward-triggered averaging (positive reward)
    plot_reward_triggered_average(
            data_pca, rewards, steps, directory=directory
    )


def plot_shaped_rewards(shaped_rewards, rewards, directory='./'):
    #steps = np.concatenate([np.arange(len(r)) for r in rewards])
    shaped_rewards = np.concatenate(shaped_rewards, axis=0)
    rewards = np.concatenate(rewards, axis=0)

    # Shaped reward relative to reward
    plt.figure()
    plt.scatter(rewards, shaped_rewards)
    plt.xlabel('Reward')
    plt.ylabel('Shaped reward')
    filename = os.path.join(directory, 'shaped-rewards.png')
    plt.savefig(filename)
    plt.close()

    # Shaped reward around reward
    window_size = 51

    data = []
    for i in range(len(shaped_rewards)):
        if rewards[i] <= 0:
            continue
        start_index = i - window_size // 2
        end_index = i + window_size // 2 + 1
        if start_index < 0:
            continue
        if end_index > len(shaped_rewards):
            continue
        
        sr = shaped_rewards[start_index:end_index]
        r = rewards[start_index-1:end_index-1] # Shaped reward is one step behind, so we shift the window on the reward by a step.
        sr[r != 0] = np.nan
        data.append(sr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        data = np.nanmean(np.stack(data), axis=0)

    plt.figure()
    x = range(-window_size//2, window_size//2)
    y = data
    plt.plot(x, y)
    plt.xlabel('Steps relative to reward')
    plt.ylabel('Shaped reward')
    plt.title('Shaped reward around positive rewards')
    plt.grid()
    filename = os.path.join(directory, 'shaped-reward-positive.png')
    plt.savefig(filename)
    print(f'Saved figure to {os.path.abspath(filename)}')

    data = []
    for i in range(len(shaped_rewards)):
        if rewards[i] >= 0:
            continue
        start_index = i - window_size // 2
        end_index = i + window_size // 2 + 1
        if start_index < 0:
            continue
        if end_index > len(shaped_rewards):
            continue
        
        sr = shaped_rewards[start_index:end_index]
        r = rewards[start_index-1:end_index-1] # Shaped reward is one step behind, so we shift the window on the reward by a step.
        sr[r != 0] = np.nan
        data.append(sr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        data = np.nanmean(np.stack(data), axis=0)

    plt.figure()
    x = range(-window_size//2, window_size//2)
    y = data
    plt.plot(x, y)
    plt.xlabel('Steps relative to reward')
    plt.ylabel('Shaped reward')
    plt.title('Shaped reward around negative rewards')
    plt.grid()
    filename = os.path.join(directory, 'shaped-reward-negative.png')
    plt.savefig(filename)
    print(f'Saved figure to {os.path.abspath(filename)}')


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default=None,
                        help='Path to results file.')
    parser.add_argument('--video-format', type=str, default=None,
                        choices=['webm', 'mp4', 'gif'],
                        help='Video format to save trajectory video in.')
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory.')
    args = parser.parse_args()

    # Read results
    results = torch.load(args.results)

    attention_data = []
    hidden_state_data = []
    reward_data = []
    shaped_reward_data = []
    for r in tqdm(results):
        attention_data.append(np.stack([
            np.concatenate([
                a[0][0].flatten(),
                a[0][1].flatten(),
                a[1][0].flatten(),
                a[1][1].flatten(),
                a[2]['action'].flatten(),
                a[2]['value'].flatten()]
            ) for a in tqdm(r['results']['attention'])
        ]))
        hidden_state_data.append(np.stack([
            np.concatenate([
                h.flatten() for h in hidden_state
            ])
            for hidden_state in r['results']['hidden']
        ]))
        reward_data.append(
            np.array(r['results']['reward'])
        )
        shaped_reward_data.append(
            np.array([
                shaped_reward.mean().item()
                for shaped_reward in r['results']['shaped_reward']
            ])
        )

    print(results[0]['results'].keys())

    # Plot results
    plot_trajectory(
            attention_data, reward_data,
            video_format=args.video_format,
            directory=args.output)

    # plot shaped rewards
    plot_shaped_rewards(shaped_reward_data, reward_data, directory=args.output)

    breakpoint()
