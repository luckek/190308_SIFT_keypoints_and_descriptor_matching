from SIFTWrapper import *
import sys


def main(argv):

    if len(argv) != 2:
        print("usage: SIFT keypoint matching: <img1> <img2>")
        sys.exit(1)

    I_1 = plt.imread(argv[0])
    I_2 = plt.imread(argv[1])

    sw = SIFTWrapper(I_1, I_2)
    kp1, kp2, good_matches = sw.compute_best_matches()

    img = cv2.drawMatchesKnn(I_1, kp1, I_2, kp2, good_matches, None, flags=2)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
