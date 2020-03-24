import cv2
import numpy as np
from imutils import build_montages
import os, re, sys
from matplotlib import pyplot as plt

def get_laplacian(image_file):
    # convert to grayscale
    gray = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2GRAY)

    height, width = gray.shape

    # create Laplacian image:
    lap = np.empty_like(gray, dtype=int)

    #for each pixel, subtract the surrounding pixel values:
    for y in range(1, height-1):
        for x in range(1, width-1):
            sum = gray[y+1,x+1]+gray[y+1,x]+gray[y-1,x+1]+gray[y-1,x]+gray[y-1,x-1]+gray[y,x-1]+gray[y+1,x-1]+gray[y+1,x]
            lap[y,x] = gray[y,x]*8 - sum

    lap = np.delete(lap, 59, 0)
    lap = np.delete(lap, 88, 1)
    lap = np.delete(lap,  0, 0)
    lap = np.delete(lap,  0, 1)

    return lap

def texture_histogram(laplacian_array):

    hist = np.histogram(laplacian_array, bins=np.arange(-2040,2041, 8))
        #cv2.calcHist([laplacian_array], [0], None, [2040], [-2040, 2040])

    """plt.figure()
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist[1][:-1],hist[0])
    plt.show()"""

    return hist[0]

def distance(hist1, hist2):

    dist_array = np.subtract(hist1, hist2)
    dist_array = np.absolute(dist_array)

    DENOM = 2 * 58 * 87
    dist = np.sum(dist_array)/DENOM

    return dist

def rank_match(query_info, targets_info):

    targets = []

    for target_name, target_info in targets_info.items():
        dist = distance(query_info, target_info)
        targets.append((target_name, dist))

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
        cv2.imwrite("texture_scores.jpg", montage)
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

    textures = {}
    texture_match = {}
    total_score = 0

    for im in images:
        textures[im] = texture_histogram(get_laplacian(im))
    for im in images:
        result = rank_match(textures[im], textures)
        target_scores = score_image(result[0][0], result[1:4], result[39], crowd_array)

        row_score = 0
        for s in target_scores[:3]:
            row_score += s[1]
        total_score += row_score
        texture_match[im] = (target_scores, row_score)

    print("TOTAL SCORE: ", total_score)
    image_display(texture_match, total_score)


if __name__ == '__main__':
   main()