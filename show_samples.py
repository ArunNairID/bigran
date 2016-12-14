
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def show_samples(samples, dim=(4,4), figsize=(10,10) ):
    fig = plt.figure(figsize=figsize)
    gs1 = gridspec.GridSpec(dim[0], dim[1])
    gs1.update(wspace=0.025, hspace=0.025)  # set the spacing between axes.

    n = samples.shape[0]
    for i in range(n):
        ax = plt.subplot(gs1[i])
        img = samples[i,:]
        img = img.reshape((28,28))
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    return fig
