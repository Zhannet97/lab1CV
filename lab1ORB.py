
import cv2, glob, time


parfumeImage = cv2.imread("../images good/parfume.JPG")
parfumeImage = cv2.cvtColor(parfumeImage , cv2.COLOR_BGR2GRAY)
files = glob.glob ("../images good/*.JPG") #good photos
#files = glob.glob ("../images bad/*.JPG") #bad photos
result = open("result.txt", "w+")

detector = cv2.ORB_create()
(kpsParfume, descsParfume) = detector.detectAndCompute(parfumeImage , None)
# result.write("PARFUME_IMAGE:\nkey points: {}, descriptors: {}\n\n".format(len(kpsParfume), descsParfume.shape));

for myFile in files:
    processTime = time.time()
    print("Processing: " + myFile)
    image = cv2.imread(myFile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (kps, descs) = detector.detectAndCompute(image, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descsParfume,descs, 2)

    goodMatch = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            goodMatch.append([m])

    matchImage = cv2.drawMatchesKnn(parfumeImage, kpsParfume, image, kps, goodMatch[1:20], None, 2)

    # cv2.imwrite(myFile + "_result"+ ".JPG", matchImage)
    result.write("{} {} {}\n".format(len(kps), len(goodMatch), time.time() - processTime))

result.close()