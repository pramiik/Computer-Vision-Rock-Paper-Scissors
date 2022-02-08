import cv2
from keras.models import load_model
import numpy as np
import random 
model = load_model('keras_model.h5')
cap = cv2.VideoCapture(0)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
player_points = 0
computer_points = 0

while True: 
    ret, frame = cap.read()
    cv2.putText(frame,'Player vs Computer', (52,457),cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0))
    resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
    data[0] = normalized_image
    prediction = model.predict(data)
    cv2.imshow('frame', frame)

    
    # Press q to close the window
    print(prediction)

    rock1= prediction[0][0] > 0.7
    paper1= prediction[0][1] > 0.7
    scissors1 = prediction[0][2] > 0.7
    nothing1 = prediction[0][3] > 0.7 

    prediction1=[rock1, paper1, scissors1, nothing1]

    game_options= [ 'rock', 'paper', 'scissors']
    computer_input = random.choice(game_options)
    print (computer_input)

    if prediction1[0][0] == True:
        player_input = ('rock')
    elif prediction1[0][1] == True:
        player_input = ('paper')
    elif prediction1[0][2] == True:
        player_input = ('scissors')
    else:
        player_input = ('nothing')

    if (player_input == rock1 and computer_input ==  'paper')  or (player_input ==paper1 and computer_input== 'scissors')  or (player_input == scissors1 and computer_input == 'paper') :
        print('player wins')
        player_points +=1

    elif (player_input ==rock1 and computer_input == 'rock')  or (player_input ==paper1 and computer_input== 'paper')  or (player_input == scissors1 and computer_input ==  'scissors') :
        print('Draw')

    elif (player_input ==paper1 and computer_input == 'rock')  or (player_input ==rock1 and computer_input == 'scissors')  or (player_input == scissors1 and computer_input == 'rock') :
        print('computer wins')
        computer_points +=1

    else :
        print('try again')

    print(f"player points : {player_points}")
    print(f"computer points : {computer_points}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()

#%%

# import the time module
import time

# define the countdown func.

	
	

# input time in seconds
t = 3

# function call
countdown(int(t))

# %%




















































'''
cv2.putText(frame,"Press"q' to quit at any time", (25, 25), Cv2.FONT _HERSHEY DUPLEX, 0.5, (150, 0, 255))
cv2.putText 
(frame,"player:". (25, 410), CV2. FONT _HERSHEY_SIMPLEX, 1, (®, 255, 255))
cv2.putText
 (frame, player_choice, (150, 410), CV2.FONT _HERSHEY _STMPLEX, 1, (8, 255, 255))
cv2.putText 
(frame,"Computer: ", (300, 418), CV2.FONT HERSHEY SIMPLEX, 1, (0, 255, 255))
cv2.putText(frame, computer choice, (490, 410), CV2.FONT HERSHEY SIMPLEX, 1, (®, 255, 255))
CV2.putText(frame, "Win:(25, 380), Cv2. FONT HERSHEY SIMPLEX, 1, (255, 255, 255))
cv2.putText(frame, str(win), (90, 300), CV2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

cv2.putText(frame, "LOss: ", (25, 325), CV2.FONT HERSHEY SIMPLEX, 1, (255, 255, 255))

CV2.putText 
(frame, str(loss), (110, 325), Cv2. FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
cv2.putText (frame,"Draw: ", (25, 350). CV2.FONT HERSHEY SIMPLEX, 1, (255, 255, 255))
cv2.putText (frame, str(draw), (115, 350), Cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))



#%%



    cv2.putText(frame,"player:", (25, 410), cv2. FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
    cv2.putText (frame,"Computer: ", (300, 418), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
    cv2.putText(frame, 
    "player points:",(25, 380), cv2. FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv2.putText(frame, "computer points: ", (25, 325), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    #cv2.putText (frame,"Draw: ", (25, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    

    
    
    cv2.putText(frame, computer_input, (490, 410), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
    cv2.putText(frame, str(win), (90, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv2.putText (frame, str(loss), (110, 325), cv2. FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv2.putText (frame, str(draw), (115, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

'''




