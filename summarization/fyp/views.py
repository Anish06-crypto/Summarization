import mimetypes
from django.http import HttpResponse
from django.shortcuts import redirect, render


import os
from os import listdir
import cv2
import glob
import numpy as np
from easyocr import Reader
import argparse
import shutil
import torchvision
import moviepy.editor as moviepy
import yake

from summarization.settings import BASE_DIR, MEDIA_ROOT, STATIC_ROOT
# from keybert import KeyBERT


# Create your views here.

def video_input(request):
    if request.method == "GET":
        return render(request,'index.html')
    if request.method == "POST":
        # Read the video from specified path
        vid =  request.FILES['video']
        cam = cv2.VideoCapture(vid.temporary_file_path())
        v_frames = []
        frames = {}
        try:
            
            # creating a folder named data
            if not os.path.exists('data'):
                os.makedirs('data')
        
        # if not created then raise error
        except OSError:
            print ('Error: Creating directory of data')
        
        # frame
        currentframe = 1
        c=1
        j=1
        count=0
        pixel_value = 0
        prev_value = 0
        while(True):
            
            # reading from frame
            ret,frame = cam.read()
            if ret:
                if c%300==0:
                    img = frame
                    data = np.asarray(img, dtype="int32")
                    pixel_value = data.sum()
                    
                    if ((pixel_value - prev_value) >= 400000) :
                        diff = pixel_value-prev_value
                        print("pi: ",pixel_value)
                        print("pr: ",prev_value)
                        print("diff: ",diff)
                        name = './data/frame' + str(currentframe) + '.jpg'
                        print ('Creating...' + name)
                        cv2.imwrite(name, frame)
                        s = "frame"+str(j)
                        frames[s] = name
                        v_frames.append(frame)
                        count+=1
                        prev_value = pixel_value
                    else:
                        continue
                c+=1
                j+=1
                currentframe += 1
            else:
                break
        print("c: ",c)
        print("j: ",j)
        # Release all space and windows once done
        cam.release()
        # cv2.destroyAllWindows()


        #easyocr detection

        def cleanup_text(text):
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV
            return "".join([c if ord(c) < 128 else "" for c in text]).strip()

        # construct the argument parser and parse the arguments
        # ap = argparse.ArgumentParser()
        # ap.add_argument("-i", "--image", required=True,help="path to input image to be OCR'd")
        # ap.add_argument("-l", "--langs", type=str, default="en",help="comma separated list of languages to OCR")
        # ap.add_argument("-g", "--gpu", type=int, default=-1,help="whether or not GPU should be used")
        # args = vars(ap.parse_args())

        # break the input languages into a comma separated list

        frame_track = {}

        j = 1
        s = ""
        for image in v_frames:
            l = []
            langs = ['en']
            print("[INFO] OCR'ing with the following languages: {}".format(langs))
            # load the input image from disk
            # image = cv2.imread(i)
            # OCR the input image using EasyOCR
            print("[INFO] OCR'ing input image...")
            reader = Reader(langs, gpu=1)
            results = reader.readtext(image)
            s = "frame"+str(j)
            j+=1
            # loop over the results
            for (bbox, text, prob) in results:
            # display the OCR'd text and associated probability
                print("[INFO] {:.4f}: {}".format(prob, text))
                # unpack the bounding box
                (tl, tr, br, bl) = bbox
                tl = (int(tl[0]), int(tl[1]))
                tr = (int(tr[0]), int(tr[1]))
                br = (int(br[0]), int(br[1]))
                bl = (int(bl[0]), int(bl[1]))
                # cleanup the text and draw the box surrounding the text along
                # with the OCR'd text itself
                text = cleanup_text(text)
                l.append(text)
                cv2.rectangle(image, tl, br, (0, 255, 0), 2)
                cv2.putText(image, text, (tl[0], tl[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # show the output image
            # cv2.imshow('processed image',image)
            ft = " ".join(l)
            frame_track[s] = ft
            ft = ""


        #video  ouput
        image_folder = 'data'
        video_name = 'summary.avi'

        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 1, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        # cv2.destroyAllWindows()
        video.release()

        
        # clip = moviepy.VideoFileClip("summary.avi")
        # clip.write_videofile("summary.mp4")

        #keyword extraction
        k = []
        language = "en"
        max_ngram_size = 3
        deduplication_threshold = 0.9
        numOfKeywords = 20
        custom_stopwords = ['the','an','a','i']
        custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, stopwords=custom_stopwords ,top=numOfKeywords, features=None)
        for key in frame_track:
            keywords = custom_kw_extractor.extract_keywords(frame_track[key])
            k.append(keywords)
            keywords =  None
        
    return render(request,'summary.html',{"kw":k,"list":frame_track.values()})


def delete_frames(request):
    files = glob.glob('data/*.jpg')
    vid_file = 'summary.avi'
    for f in files:
        os.remove(f)
    os.remove(vid_file)
    return redirect('/index')
