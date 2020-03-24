import cv2
import numpy as np
from imutils import build_montages
import os, re, sys
from matplotlib import pyplot as plt
import colordistance, texturedistance, shapedistance


def grand_ranking(colors_rank, textures_rank, shapes_rank):

    """print(colors_rank)
    print(textures_rank)
    print(shapes_rank)"""

    targets = []

    colors = {}
    textures = {}
    shapes = {}
    for i in colors_rank:
        colors[i[0]] = i[1]
    for i in textures_rank:
        textures[i[0]] = i[1]
    for i in shapes_rank:
        shapes[i[0]] = i[1]

    for target_name, color_dist in colors.items():
        text_dist = textures[target_name]
        overlap = shapes[target_name]

        #AMEE:
        if color_dist < 0.3:
            tot = 0.7*color_dist +  0.2 * overlap + 0.1*text_dist
        elif color_dist < 0.6 and text_dist < 0.6:
            tot = 0.7 * color_dist + 0.15 * text_dist + 0.25 * overlap
        else:
            tot = 0.8*color_dist +  0.2 * overlap

        """
        #CROWD:
        if color_dist < 0.3:
            tot = 0.7*color_dist +  0.2 * overlap + 0.1*text_dist
        elif color_dist < 0.6 and text_dist < 0.6:
            tot = 0.4 * color_dist + 0.1 * text_dist + 0.5 * overlap
        else:
            tot = color_dist
        """

        targets.append((target_name, tot))

    targets.sort(key=lambda x: x[1])

    # print(targets)
    return targets

def score_image(name, top3, most_unalike, crowdsource_array):
    q = int(re.sub('[^0-9]', '', name)) - 1
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

def image_display(query_to_scores):
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
        cv2.imwrite("total_scores.jpg", montage)
        cv2.waitKey(0)

def main():

    directory = 'resources/images/'
    #crowd_file = open('resources/Crowd.txt', 'r')
    crowd_file = open('resources/Amee.txt', 'r')
    crowd_array = [[int(num) for num in line.split()] for line in crowd_file]

    # Sort through the images in the directory
    images = []
    for file in os.listdir(directory):
        if file.endswith(".jpg"):
            images.append(directory + file)
    images.sort()

    image_match = {}

    colors = {}
    textures = {}
    shapes = {}

    total_score = 0

    for im in images:
        colors[im] =  colordistance.color_histogram(im)
        textures[im] = texturedistance.texture_histogram(texturedistance.get_laplacian(im))
        shapes[im] = shapedistance.binarize(im)

    for im in images:
        color_result = colordistance.rank_match(colors[im], colors)
        texture_result = texturedistance.rank_match(textures[im], textures)
        shape_result = shapedistance.rank_match(shapes[im], shapes)

        result = grand_ranking(color_result[1:], texture_result[1:], shape_result[1:])

        target_scores = score_image(im, result[0:3], result[38], crowd_array)

        row_score = 0
        for s in target_scores[:3]:
            row_score += s[1]
        total_score += row_score
        image_match[im] = (target_scores, row_score)

    print("TOTAL SCORE: ", total_score)
    image_display(image_match)




if __name__ == '__main__':
   main()