import os

import numpy as np

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg") # This prevents program failing on mac
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

import tsne # Import the tsne file
import pylab
# Similar to visualization to tsne, but is faster
import umap

# Directory of the data
base_dir = os.path.join("./flowers")


def plotDistribution():
    print("Plotting distribution")
    names = []
    counts = []
    for dir in os.listdir(base_dir):
        names.append(dir)
        counts.append(len(os.listdir(os.path.join(base_dir, dir))))
    # Plotting the counts
    plt.bar(names, counts)
    plt.show()

def showImages():
    print("Showing images")
    columns = 10
    rows = len(os.listdir(base_dir))

    fig=plt.figure() #(figsize=(8,8))
    i = 0
    for dir in os.listdir(base_dir): # this is one row
        j = 0
        dirPath = os.path.join(base_dir, dir)
        for imgName in os.listdir(dirPath): # the columns
            if j >= columns: # only show first 10 images
                break
            # img = np.random.randn(100,100)
            # img = mpimg.imread(os.path.join(dirPath, imgName))
            img = Image.open(os.path.join(dirPath, imgName))
            img.thumbnail((64, 64), Image.ANTIALIAS) # resizes image in-place
            axis = fig.add_subplot(rows, columns, i*columns + j + 1)
            axis.set_axis_off()
            axis.set_title(dir + "_" + str(j))
            plt.imshow(img)
            j += 1
        i += 1

    plt.show()

# THIS DOES NOT WORK YET.
# The dimensions of the data being passed to t-SNE is too high.
# It needs to be reduced, otherwise it is too complex and runs out of memory.
def runTsne():
    size = 200

    labels = np.array([])
    images = np.array([]).reshape(0,size*size*3) # 3 for the colour channels

    i = 0
    for dir in os.listdir(base_dir): # this is one row
        j = 0
        dirPath = os.path.join(base_dir, dir)
        for imgName in os.listdir(dirPath): # the columns
            if j >= 10: # Only use 100 images
                break
            img = Image.open(os.path.join(dirPath, imgName))
            img = img.resize((size,size), Image.ANTIALIAS)
            np_img = np.array(img)
            np_img = np_img.reshape(1,-1)

            # print(np_img.shape)
            # print(labels.shape)
            # print(images.shape)
            labels = np.append(labels,dir)
            images = np.concatenate((images, np_img))
            print(images.shape)
            j += 1
        i += 1

    print("Running Tsne on " + str(len(labels)) + " data points")
    print(images.shape)
    Y = tsne.tsne(images, 2, 50, 30)

    pylab.scatter(Y[:,0], Y[:,1], 20, labels)
    pylab.show()
    # fig, ax = plt.subplots()
    # for name in os.listdir(base_dir):
    #     colour = np.random.rand(1)
    #     x = Y[:, 0]
    #     y = Y[:, 1]
    #     ax.scatter(x, y, c=colour, label=name)
    #
    # ax.legend()
    # plt.show()


def runPCA():
    print("Running PCA")

def runUMAP():
    size = 200
    name_dict = {}
    labels = np.array([])
    images = np.array([]).reshape(0,size*size*3) # 3 for the colour channels
    i = 0
    for dir in os.listdir(base_dir): # this is one row
        j = 0
        dirPath = os.path.join(base_dir, dir)
        name_dict[i] = dir
        for imgName in os.listdir(dirPath): # the columns
            if j >= 100: # Only use 100 images
                break
            img = Image.open(os.path.join(dirPath, imgName))
            img = img.resize((size,size), Image.ANTIALIAS)
            np_img = np.array(img)
            np_img = np_img.reshape(1,-1)

            # labels = np.append(labels,dir)
            labels = np.append(labels,i)
            images = np.concatenate((images, np_img))
            j += 1
        i += 1

    print("Running UMAP on " + str(len(labels)) + " data points")

    reducer = umap.UMAP(
        n_neighbors=10,
        min_dist=0.1,
        metric='correlation')
    reducer.fit(images)
    embedding = reducer.transform(images)
    print(embedding.shape)

    # Create plot
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral')#, s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(i+1)-0.5).set_ticks(np.arange(i+1))
    plt.show()






def main():
    # plotDistribution()
    # showImages()
    # runTsne()
    runUMAP()

main()


# EOF
