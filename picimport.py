import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def getpic(name):
    I = mpimg.imread(name)   #h,w,c
    return I
    # for
    # I =
    # return I


def main():
    # type = '1'
    # num = '145'
    # name = './' + type + '-' + num + '.jpg'
    # try:
    #     P = getpic(name)
    # except:
    #     print('import pic:' + name + ' fail')
    # else:
    #     print('import pic:' + name + ' success')
    #     print(P.shape)
    #
    # print(P)
    # plt.imshow(P)
    # plt.show()

    type_tigerI = ['tigerI_1', 'tigerI_2', 'tigerI_3', 'tigerI_4', 'tigerI_5']
    type_tigerII = ['tigerII_1', 'tigerII_2', 'tigerII_3', 'tigerII_4', 'tigerII_5']
    name1 = list(range(5))
    name2 = list(range(5))
    P1 = list(range(5))
    P2 = list(range(5))
    count = 0
    for i in range(len(name1)):
        count += 1
        name1[i] = './Test Pictures/' + type_tigerI[i] + '.jpg'
        name2[i] = './Test Pictures/' + type_tigerII[i] + '.jpg'
        # name1 = './Test Pictures/' + type_tigerI + '.jpg'
        # name2 = './Test Pictures/' + type_tigerII + '.jpg'
    for i in range(len(name1)):
        P1[i] = getpic(name1[i])
        P2[i] = getpic(name2[i])
        # plt.imshow(P1[i])
        # plt.show()
        # plt.imshow(P2[i])
        # plt.show()
        print(P1[i])


if __name__ == '__main__':
    main()



# from PIL import Image
# import numpy as np
#
# I = Image.open('./Test Pictures/tigerI_1.jpg')
# # I.show()
# # I.save('./save.png')
# I_array = np.array(I)
# print(I_array.shape)



# import matplotlib.pyplot as plt
# from scipy import misc
# import scipy
# I = misc.imread('./Test Pictures/tigerI_1.jpg')
# # scipy.misc.imsave('./save1.png', I)
# plt.imshow(I)
# plt.show()
# print(I.shape)
