# Original Guide
# https://medium.com/better-programming/beginners-guide-to-tesseract-ocr-using-python-10ecbb426c3d

# Thesseract materials: https://github.com/tesseract-ocr/tessdoc
# Improving quality: https://tesseract-ocr.github.io/tessdoc/ImproveQuality



from tesserocr import PyTessBaseAPI, RIL, PSM
from PIL import Image


# testing files
scanned_file = 'scan_document.png'
receipt_easy = 'receipt_easy.jpeg'
receipt_hard = 'receipt_hard.jpg'
license_plate = 'license_plate.jpg'

images = ['license_plate.jpg', 'receipt_example.jpg', 'scan_document.png']



# BASELINE OPTIONS RUN - Scanned file

image = Image.open(scanned_file)

# text and confidence value for each word
with PyTessBaseAPI() as api:
    api.SetImage(image)
    scan_text = api.GetUTF8Text()
    confidence_values = api.AllWordConfidences()

print(scan_text)
print(confidence_values)


# Print boxes
with PyTessBaseAPI() as api:
    api.SetImage(image)
    boxes = api.GetComponentImages(RIL.TEXTLINE, True)
    print('Found {} textline image components.'.format(len(boxes)))
    for i, (im, box, _, _) in enumerate(boxes):
        # im is a PIL image object
        # box is a dict with x, y, w and h keys
        api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
        ocrResult = api.GetUTF8Text()
        conf = api.MeanTextConf()
        print(u"Box[{0}]: x={x}, y={y}, w={w}, h={h}, "
              "confidence: {1}, text: {2}".format(i, conf, ocrResult, **box))


# Receipt Easy
image = Image.open(receipt_easy)

gray = image.convert('L')
blackwhite = gray.point(lambda x: 0 if x < 200 else 255, '1')
# blackwhite.save("receipt_easy_bw.jpg")

with PyTessBaseAPI(psm=PSM.AUTO_OSD) as api:
    api.SetImage(blackwhite)
    scan_text = api.GetUTF8Text()
    confidence_values = api.AllWordConfidences()

print(scan_text)
print(confidence_values)



# Receipt Hard
image = Image.open(receipt_hard)

gray = image.convert('L')
blackwhite = gray.point(lambda x: 0 if x < 200 else 255, '1')
# blackwhite.save("receipt_easy_bw.jpg")

with PyTessBaseAPI(psm=PSM.AUTO_OSD) as api:
    api.SetImage(blackwhite)
    scan_text = api.GetUTF8Text()
    confidence_values = api.AllWordConfidences()

print(scan_text)
print(confidence_values)



# License Plate 1 - better but still far from ideal
license_plate2 = 'license_plate2.jpg'
image = Image.open(license_plate2)

gray = image.convert('L')
blackwhite = gray.point(lambda x: 0 if x < 200 else 255, '1')
# blackwhite.save("receipt_easy_bw.jpg")

with PyTessBaseAPI(psm=PSM.SINGLE_LINE) as api:
    api.SetImage(blackwhite)
    scan_text = api.GetUTF8Text()
    confidence_values = api.AllWordConfidences()

print(scan_text)
print(confidence_values)


# License Plate 2 - 
image = Image.open(license_plate)

gray = image.convert('L')
blackwhite = gray.point(lambda x: 0 if x < 200 else 255, '1')
# blackwhite.save("receipt_easy_bw.jpg")

with PyTessBaseAPI(psm=PSM.SINGLE_BLOCK_VERT_TEXT) as api:
    api.SetImage(blackwhite)
    scan_text = api.GetUTF8Text()
    confidence_values = api.AllWordConfidences()

print(scan_text)
print(confidence_values)


# multiple images
with PyTessBaseAPI() as api:
    for img in images:
        api.SetImageFile(img)
        print(api.GetUTF8Text())






