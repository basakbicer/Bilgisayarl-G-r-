# -*- coding: utf-8 -*-
"""
Created on Mon May 13 21:17:09 2024

@author: Casper
"""

import cv2 as cv
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils  #mediapipe kütüphanesinden, çizim yardımcı işlevlerini ve el tespiti modülünü içe aktarır.
mp_hands = mp.solutions.hands


cap = cv.VideoCapture(0) #işlevi, bilgisayarınızda mevcut olan bir kameradan video akışını yakalamak için bir VideoCapture nesnesi oluşturur.

with mp_hands.Hands(       #sınıfını kullanarak bir el tespit örneği oluşturur. Bu, el tespiti için belirli yapılandırmaları ayarlar.
    model_complexity=0,
    min_detection_confidence=0.5, #En az %50 güvenle tespit edilen eller işlenir.
    min_tracking_confidence=0.5) as hands: #En az %50 güvenle takip edilen eller işlenir.
  while cap.isOpened(): #Video akışının açık olduğu sürece bir döngü başlatır.
    success, img = cap.read()  #cap.read() işlevi, bir sonraki video çerçevesini yakalar ve bu çerçeveyi img değişkenine atar. success değişkeni, çerçevenin başarıyla alınıp alınmadığını belirtir. Eğer alınamazsa, döngüyü sonlandırır.
    if not success:
      break

    img.flags.writeable = False   #Çerçevenin yazılabilirliğini kapatır, ardından çerçeve formatını BGR'den RGB'ye dönüştürür. Daha sonra, mediapipe.Hands nesnesinin process() yöntemini kullanarak el tespiti işlemini gerçekleştirir.
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(img)

    img.flags.writeable = True   #El tespit işlemi tamamlandıktan sonra, çerçeve formatını tekrar RGB'den BGR'ye dönüştürür ve çerçevenin yazılabilirliğini açar.
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    counter = 0  #Parmak sayısını saklamak için bir sayaç değişkeni oluşturu

    if results.multi_hand_landmarks:  #Eğer el tespiti sonuçları el tespiti yapıldığını gösteriyorsa, yani en az bir el bulunduysa, bu bloğa girer.

      for hand_landmarks in results.multi_hand_landmarks:  #Her bir elin çizgilerini (landmarks) içeren sonuçları döngüyle işler.
        # Check left and right
        handIndex = results.multi_hand_landmarks.index(hand_landmarks) # Tespit edilen eller listesindeki mevcut elin indeksini alır.
        handLabel = results.multi_handedness[handIndex].classification[0].label #Elin sağ mı sol mu olduğunu belirten etiketi alır. multi_handedness, tespit edilen ellerin hangi elde olduğunu (Right veya Left) belirler.

        handLandmarks = []  #El üzerindeki her bir landmark'ın (noktanın) koordinatlarını elde eder ve bunları bir listeye ekler.
        
        for landmarks in hand_landmarks.landmark: 
          handLandmarks.append([landmarks.x, landmarks.y])

        if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]: #Elin sağ veya sol olduğunu kontrol eder ve parmakların açık olduğunu kontrol eder.
          counter = counter+1
        elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]: #Baş parmak açık mı?
          counter = counter+1

        if handLandmarks[8][1] < handLandmarks[6][1]:       # İşaret parmağı . Eldeki işaretçi, orta, yüzük ve serçe parmaklarının yukarı yönlü olup olmadığını kontrol eder ve parmak sayısını artırır.
          counter = counter+1
        if handLandmarks[12][1] < handLandmarks[10][1]:     #Orta parmak 
          counter = counter+1
        if handLandmarks[16][1] < handLandmarks[14][1]:    #Yüzük parmağı 
          counter = counter+1
        if handLandmarks[20][1] < handLandmarks[18][1]:    #Serçe parmak 
          counter = counter+1

        # Draw hand
        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS) #El üzerindeki landmark'ları ve el bağlantılarını (joint connections) görüntüye çizer.


    cv.putText(img, str(counter), (50, 100), cv.FONT_HERSHEY_SIMPLEX, 3, (255,0,255), 10) #Elde bulunan parmak sayısını çerçevenin üst sol köşesine metin olarak ekler.
    cv.imshow('Hand Detection', img) #İşlenmiş çerçeveyi görüntüler.
    if cv.waitKey(5) & 0xFF == 27: #Klavyeden ESC tuşuna basıldığında döngüyü kırar ve programı kapatır.
      break
cap.release()