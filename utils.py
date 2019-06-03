import skimage
import skimage.io
import skimage.transform
import numpy as np
import cv2
import matplotlib.pyplot as plt


# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))

def save_feature(img_datas,img_name):
    """
    img_datas: [1,H,W,C]
    img_name: such as conv1_1
    """
    img_datas = np.array(img_datas)
    img_datas = np.squeeze(img_datas)

    img_num = img_datas.shape[2]
    squr = img_num ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    for i in range(img_num):
        img_data = img_datas[:,:,i]
        # plt.subplot(row,col,i+1)
        plt.imshow(img_data)
        plt.savefig('feature_pic/'+img_name+'_%d.jpg'%i)
        # axis('off')
        # title('feature map: %d'%i)
    # plt.savefig('feature_pic/'+img_name+'.jpg')
    # plt.show()
    all_add = sum([img_datas[:,:,i] for i in range(img_num)])
    plt.imshow(all_add)
    plt.savefig('feature_pic/'+img_name+'add.jpg')
def test():
    img = skimage.io.imread("./test_data/puzzle.jpeg")
    ny = 300
    nx = int(img.shape[1] * ny / img.shape[0])
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/output.jpeg", img)


if __name__ == "__main__":
    test()
