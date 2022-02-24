from PIL import Image

img_1 = Image.open("IMG_1.jpg")
img_2 = Image.open("IMG_2.jpg")
img_3 = Image.open("IMG_3.jpg")
img_4 = Image.open("IMG_4.jpg")
img_5 = Image.open("IMG_5.jpg")

imgSize = img_1.size

#w*h

#new_im = Image.new('RGB', (2*img_01_size[0],2*img_01_size[1]), (250,250,250))
#im = PIL.Image.new(mode="RGB", size=(200, 200))

#empty image
mergedImg = Image.new(mode="RGB", size=(3*imgSize[0], 2*imgSize[1]), color=(250,250,250))

mergedImg.paste(img_1, (0,0))
mergedImg.paste(img_2, (imgSize[0],0))
mergedImg.paste(img_3, (imgSize[0]*2,0))
mergedImg.paste(img_4, (0,imgSize[1]))
mergedImg.paste(img_5, (imgSize[0],imgSize[1]))

mergedImg.save("mergedImage.jpg")
mergedImg.show()

