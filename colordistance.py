import cv2
import numpy as np
from imutils import build_montages
import os, re


#1 Generate each image’s three dimensional color histogram,
# and then compare the histograms using the “normalized L1 distance”
def color_histogram(filename):

    im = cv2.imread(filename)

    # number of bins for each color: 16 * 8 * 8 = 1024 total bins (colors)
    REDS = 12
    GREENS = 12
    BLUES = 6

    image = [im]
    channels = [0, 1, 2]
    bins = [REDS, GREENS, BLUES]
    ranges = [0, 256, 0, 256, 0, 256]

    # histogram calculation
    hist = cv2.calcHist(image, channels, None, bins, ranges)
    hist = hist.astype(int)

    #np.set_printoptions(threshold=sys.maxsize)
    #print("HIST FOR ", filename)
    #print(hist)

    return hist

#2 Find the L1 distance : L1(Image1, Image2) = 􏰀 SUM r,g,b (|Image1(r, g, b) − Image2(r, g, b)| /(2 ∗ rows ∗ cols))
def l1_distance(hist1, hist2):

    dist_array = np.subtract(hist1, hist2)
    dist_array = np.absolute(dist_array)

    #print("DIST ARRAY: ")
    #print(dist_array)

    DENOM = 2 * 60 * 89
    dist = np.sum(dist_array)/DENOM

    #print("DISTANCE = ", dist)
    return dist

def rank_match(query_info, targets_info):

    targets = []

    for target_name, target_info in targets_info.items():
        dist = l1_distance(query_info, target_info)
        targets.append((target_name, dist))

    targets.sort(key=lambda x: x[1])

    #print(targets)
    return targets

#3 For each of the 40 query images, find the three target images most like it in color distribution,
# that is, the three target images that have the smallest L1 distance to the query image

#4 Score each query image q
# Score(q) = Crowd(q, t1) + Crowd(q, t2) + Crowd(q, t3)
def score_image(query, top3, most_unalike, crowdsource_array):

    q = int(re.sub('[^0-9]', '', query)) - 1
    num40 = int(re.sub('[^0-9]', '', most_unalike[0])) - 1

    indices = []
    #top3 is a list of 3 tuples (filename, distance)
    for target in top3:
        index = re.sub('[^0-9]', '', target[0])
        indices.append((target[0],int(index) - 1))
    indices.append((most_unalike[0],num40))

    # PERSONAL SCORES
    aa = []
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
    #print(sum(counts), "+")

    #CROWD SOURCE SCORES
    scores = []
    for t in indices:
        scores.append((t[0], crowdsource_array[q][t[1]]))

    print(scores)
    return scores

def image_display(query_to_scores, total_score):
    # 40 by 5 image display of q, t1, t2, t3, t40, plus individual target scores, row scores, and grand total score

    blank = cv2.imread('resources/blank.jpg')
    images = []

    for query in query_to_scores.keys():
        """i = 1
        blank = cv2.imread('resources/blank.jpg')
        cv2.putText(blank, str(i), (5, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        images.append(blank)
        i+=1"""

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
        cv2.imwrite("color_scores.jpg", montage)
        cv2.waitKey(0)



def main():
    directory = 'resources/images/'
    crowd_file = open('resources/Crowd.txt', 'r')
    crowd_array = [[int(num) for num in line.split()] for line in crowd_file]

    images = []
    for file in os.listdir(directory):
        if file.endswith(".jpg"):
            images.append(directory+file)
    images.sort()

    colors = {}
    color_match = {}
    total_score = 0

    for im in images:
        colors[im] = color_histogram(im)
    for im in images:
        result = rank_match(colors[im], colors)
        target_scores = score_image(result[0][0], result[1:4], result[39], crowd_array)

        row_score = 0
        for s in target_scores[:3]:
            row_score += s[1]
        total_score += row_score
        color_match[im] = (target_scores, row_score)

    print("TOTAL SCORE: ", total_score)
    image_display(color_match, total_score)


if __name__ == '__main__':
   main()