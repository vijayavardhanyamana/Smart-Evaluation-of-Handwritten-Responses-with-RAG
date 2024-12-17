from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# import requests
# from PIL import Image

# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

save_directory = "models/trocr_base_handwritten"

# processor.save_pretrained(save_directory)
# model.save_pretrained(save_directory)


# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
# model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

# save_directory = "trocr-large-handwritten"

# processor.save_pretrained(save_directory)
# model.save_pretrained(save_directory)


processor = TrOCRProcessor.from_pretrained(save_directory)
model = VisionEncoderDecoderModel.from_pretrained(save_directory)

def text(image_path):
    t = ""
    for i in image_path:
        image = Image.open(i).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        t = t+generated_text.replace(" ", "")+ " "
        
    # print(t)
        
    return t
    