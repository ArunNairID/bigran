
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def show_autoencoding(x, ys, figsize=(10,10) ):
    n = x.shape[0]
    m = ys.shape[0] / n

    fig = plt.figure(figsize=figsize)
    gs1 = gridspec.GridSpec(n, m+1)
    gs1.update(wspace=0.025, hspace=0.025)  # set the spacing between axes.

    for i in range(n):
        ax = plt.subplot(gs1[i,0])
        img = x[i,0,:,:]
        ax.imshow(img, cmap='gray')
        ax.axis('on')
        ax.spines['bottom'].set_color('red')
        ax.spines['top'].set_color('red')
        ax.spines['right'].set_color('red')
        ax.spines['left'].set_color('red')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        for j in range(m):
            ax = plt.subplot(gs1[i,j+1])
            img = ys[i*m+j,0,:,:]
            ax.imshow(img, cmap='gray')
            ax.axis('off')
    return fig
