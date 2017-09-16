import cv2

basedir = 'my-examples/'

image_file = basedir + 'normal.jpg'
img = cv2.imread(image_file)
flipped = cv2.flip(img, 1)
cv2.imwrite(basedir + 'flipped.jpg', flipped)



