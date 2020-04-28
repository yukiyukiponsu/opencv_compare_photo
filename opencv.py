from PIL import Image
import cv2

pathX_1 = "Img/img1_1"
# path = "Img/img2_1"
imgX_1 = cv2.imread(pathX_1) #画像の読み込み
print("画像1の大きさ")
print(imgX_1.shape) #画像の大きさの表示

imgX_1_resize = cv2.resize(imgX_1, dsize=(100, 100)) #画像の大きさを100×100にする
print("画像1の大きさを100×100に変更")
print(imgX_1_resize.shape) 

#画像を色相分布行列化する = ヒストグラム化する
#このサイトがわかりやすい（http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_histograms/py_histogram_begins/py_histogram_begins.html）
target_hist = cv2.calcHist([imgX_1_resize], [0], None, [256], [0, 256])

pathX_2 = "Img/img1_2"
#pathX_2 = "Img/img2_2"
imgX_2 = cv2.imread(pathX_2)
print("画像2の大きさ")
print(imgX_2.shape)

imgX_2_resize = cv2.resize(imgX_2, dsize=(100, 100)) #画像の大きさを100×100にする
print("画像2の大きさを100×100に変更")
print(imgX_2_resize.shape) 
compare_hist = cv2.calcHist([imgX_2_resize], [0], None, [256], [0, 256])

#画像の比較結果
#compareHistについて詳しく知りたい方（http://opencv.jp/opencv-2svn/cpp/imgproc_histograms.html）
#imgX_1とimgX_2のヒストグラムを比較
ret = cv2.compareHist(target_hist, compare_hist, 0)
print(ret)

#今回のプログラムでは画像の一致をある程度検出できればいいので、retの値が99%以上の適合率であればよしとしています
if ret > 0.99:
    print("match!!")
else:
    print("not match..")