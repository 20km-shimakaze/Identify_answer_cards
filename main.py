import cv2
import numpy as np


def cv_show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sort_counts(cnts, method='left-right'):
    revers = False
    i = 0
    if method == 'right-left' or method == 'bottom-top':
        revers = True
    if method == 'bottom-top' or method == 'top-bottom':
        i = 1
    bounding_box = [cv2.boundingRect(c) for c in cnts]
    (cnts, bounding_box) = zip(*sorted(zip(cnts, bounding_box), key=lambda x: x[1][i], reverse=revers))
    return cnts, bounding_box


def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def point_len(x, y):
    return np.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))


def four_points_transform(img, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect
    width = int(max(point_len(tl, tr), point_len(bl, br)))
    height = int(max(point_len(tl, bl), point_len(tr, br)))
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype='float32')
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (width, height))
    return warp


answer = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

img_bgr = cv2.imread('images/test_01.png')
img_copy = img_bgr.copy()
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
edge = cv2.Canny(img_gray, 75, 200)
# cv_show(edge)
counts = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
# cv_show(cv2.drawContours(img_copy, counts, -1, (0, 0, 255), 1))
doc_cnt = None
if len(counts):
    counts = sorted(counts, key=cv2.contourArea, reverse=True)
    # 近似轮廓
    for c in counts:
        # 轮廓 精度 封闭
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            doc_cnt = approx
            break
# 变换
warped = four_points_transform(img_gray, doc_cnt.reshape(4, 2))
img_binary = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# cv_show(img_binary)
counts = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
question_counts = []
# warped = cv2.drawContours(warped, counts, -1, (0, 0, 255), 2)
# cv_show(warped)
for cnt in counts:
    x, y, w, h = cv2.boundingRect(cnt)
    ar = w / float(h)
    if 0.9 <= ar <= 1.1 and w >= 20 and h >= 20:
        question_counts.append(cnt)
question_counts = sort_counts(question_counts, 'top-bottom')[0]
for q, i in enumerate(np.arange(0, len(question_counts), 5)):
    cnts = sort_counts(question_counts[i:i + 5])[0]
    ans = None
    for num, cnt in enumerate(cnts):
        mask = np.zeros(img_binary.shape, dtype='uint8')
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mask = cv2.bitwise_and(img_binary, img_binary, mask=mask)
        total = cv2.countNonZero(mask)
        if ans is None or total > ans[1]:
            ans = (num, total)
    warped = cv2.drawContours(warped, [cnts[ans[0]]], 0, (0, 0, 255), 2)
    print('第', q + 1, '行填写为', chr(ans[0]+65))
cv_show(warped)
