import cv2, time
import pandas as pd


# CAPTURING IMAGE
# cascade_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# img = cv2.imread("shushkov(0).jpg", 1)
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = cascade_classifier.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors = 5)
# for x, y, w, h in faces:
#     img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

# CAPTURING VIDEO FIRST CADR
# video = cv2.VideoCapture(0)
# check, frame = video.read() # frame - is the first image in video
# time.sleep(3)
# cv2.imshow("capturing", frame)
# cv2.waitKey(0)
# video.reliase()
# cv2.destroyAllWindows()

# CAPTURING THE VIDEO INSTEAD OF FIRST FRAME
first_name = None
status_list = [None, None]
times = []
df = pd.DataFrame(columns = ["Start", "End"])
video = cv2.VideoCapture(0)

# a = 1

while True:
#     a += 1
    check, frame = video.read()
    status = 0
#     print(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_name is None:
        first_frame = gray
        continue
    status = 1

    delta_frame = cv2.absdiff(first_frame, gray)
    thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta, None, iterations = 0)

    (_, cnts, _) = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    status_list.append(status)

    status_list = status_list[-2:]

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(detetime.now())

    cv2.imshow("frame", frame)
    cv2.imshow("capturing", gray)
    cv2.imshow("delta", delta_frame)
    cv2.imshow("thresh", thresh_delta)

#     cv2.imshow("Capturing", gray)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

# print(a)
print(status_list)
print(times)

for i in range(0, len(times), 2):
    df = df.append({"Start": times[i], "End": times[i + 1]}, ignore_index = True)

df.to_cvs("Times.csv")

video.release()
cv2.destroyAllWindows()

# USE CASE - MOTION DETECTOR


