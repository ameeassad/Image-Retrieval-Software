import cv2
import numpy as np
from imutils import build_montages
import os, re, sys
from matplotlib import pyplot as plt

def binarize(image_file):
    # convert to grayscale
    image = cv2.imread(image_file)

    height, width,_ = image.shape

    # create Laplacian image:
    binarized = np.empty([height,width], dtype=int)

    #for each pixel
    for y in range(height):
        for x in range(width):
            if (image[y, x, 0]<30 and image[y, x, 1]<30 and image[y, x, 2]<30):
                # BLACK
                binarized[y,x] = 0
            else:
                # WHITE
                binarized[y,x] = 1

    return binarized

def normalized_overlap(binary1, binary2):

    #􏰂 SUM (x,y) |ImageN1(x, y) ̸= ImageN2(x, y)| /(rows ∗ cols)

    overlap = np.equal(binary1, binary2)
    sum = np.count_nonzero( overlap == False)

    result = sum/(60*89)
    return result

def rank_match(query_info, targets_info):

    targets = []

    for target_name, target_info in targets_info.items():
        overlap = normalized_overlap(query_info, target_info)
        targets.append((target_name, overlap))

    targets.sort(key=lambda x: x[1])

    #print(targets)
    return targets

def score_image(query, top3, most_unalike, crowdsource_array):
    q = int(re.sub('[^0-9]', '', query)) - 1
    num40 = int(re.sub('[^0-9]', '', most_unalike[0])) - 1

    indices = []
    #top3 is a list of 3 tuples (filename, distance)
    for target in top3:
        index = re.sub('[^0-9]', '', target[0])
        indices.append((target[0],int(index) - 1))
    indices.append((most_unalike[0],num40))

    aa = []
    # PERSONAL SCORES
    for a in indices[:-1]:
        aa.append(a[1]+1)
    counts = []
    with open("resources/MyPreferences.txt", "r") as mypref:
        for i, line in enumerate(mypref):
            if i == q:
                count = 0
                line = line.split()[1:]
                for l in line:
                    if int(l) in aa:
                        count +=1
                counts.append(count)

    #print("Personal match for", q+1, ":", sum(counts))
    print(sum(counts), "+")

    scores = []
    for t in indices:
        scores.append((t[0], crowdsource_array[q][t[1]]))
    return scores

def image_display(query_to_scores, total_score):
    # 40 by 5 image display of q, t1, t2, t3, t40, plus individual target scores, row scores, and grand total score

    images = []

    for query in query_to_scores.keys():

        targets = query_to_scores[query][0]
        row_score = query_to_scores[query][1]

        q_image = cv2.imread(query)
        cv2.putText(q_image, str(row_score), (5, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        images.append(q_image)

        for target in targets:
            t_image = cv2.imread(target[0])
            cv2.putText(t_image, str(target[1]), (5,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            images.append(t_image)

    montages = build_montages(images, (89,60), (5, 40))

    # loop over the montages and display each of them
    for montage in montages:
        cv2.imshow("Montage", montage)
        cv2.imwrite("shape_scores.jpg", montage)
        cv2.waitKey(0)

def main():

    directory = 'resources/images/'
    crowd_file = open('resources/Crowd.txt', 'r')
    crowd_array = [[int(num) for num in line.split()] for line in crowd_file]

    images = []
    for file in os.listdir(directory):
        if file.endswith(".jpg"):
            images.append(directory + file)
    images.sort()

    shapes = {}
    shape_match = {}
    total_score = 0

    for im in images:
        shapes[im] = binarize(im)
    for im in images:
        result = rank_match(shapes[im], shapes)
        target_scores = score_image(result[0][0], result[1:4], result[39], crowd_array)

        row_score = 0
        for s in target_scores[:3]:
            row_score += s[1]
        total_score += row_score
        shape_match[im] = (target_scores, row_score)

    print("TOTAL SCORE: ", total_score)
    image_display(shape_match, total_score)


if __name__ == '__main__':
   main()