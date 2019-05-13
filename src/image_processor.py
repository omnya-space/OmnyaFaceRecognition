import cv2
import numpy as np

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 4)




def draw_text(img, text, x, y,font_size,thickness):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, font_size, (0, 255, 0), thickness)





def label_image(image, name, rect):
    draw_rectangle(image, rect)
    font_size = 1
    thickness = 2
    draw_text(image, name, rect[0], rect[1] - 5, font_size, thickness)
    return image



