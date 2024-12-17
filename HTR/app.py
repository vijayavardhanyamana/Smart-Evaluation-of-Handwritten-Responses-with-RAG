import cv2


from HTR.word import convert_image
from HTR.strike import struck_images
from HTR.hcr import text
from HTR.spell_and_gramer_check import spell_grammer

# Define a function to extract text from an image
def extract_text_from_image(img_path):
    img = cv2.imread(img_path)
    # print(img)
    convert_image(img)
    images_path = struck_images()
    t = text(images_path)
    # print("\n\n\n\n\n\n\n")
    # print(t)
    t = spell_grammer(t)
    # t = text
    # print("\n\n\n\n\n\n\n")
    # print(t)
    return t

# extract_text_from_image("ans_image/1.jpg")


    




