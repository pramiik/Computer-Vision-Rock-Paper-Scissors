
#%%

import random

import cv2 
from keras.models import load_model
import numpy as np
import random 
model = load_model('keras_model.h5')
cap = cv2.VideoCapture(0)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

while True: 
    ret, frame = cap.read()
    cv2.putText(frame,'Player vs Computer', (52,457),cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 5, cv2.LINE_4)
    resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
    data[0] = normalized_image
    prediction = model.predict(data)
    cv2.imshow('frame', frame)
    # Press q to close the window
    rock1= prediction[0][0] > 0.7
    paper1= prediction[0][1] > 0.7
    scissors1 = prediction[0][2] > 0.7
    nothing1 = prediction[0][3] > 0.7 
    
    prediction1=[rock1, paper1, scissors1, nothing1]

    player_input =True
#get the random choice from computer 
    game_options= [ 'rock', 'paper', 'scissors']
    computer_input = random.choice(game_options)

    if prediction1[0] == True:
        player_input == ('rock')
    elif prediction1[1] == True:
        player_input == ('paper')
    elif prediction1[2] == True:
        player_input == ('scissors')
    else:
        player_input == ('nothing')

    cv2.putText(frame,str(player_input), (52,244),cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 5, cv2.LINE_4)
    print(prediction1)

    print(computer_input)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()


#%%

#open the camera
#get the player input 
rock= prediction[0][0] > 0.7
paper = prediction[0][1] > 0.7
scissors = prediction[0][2] > 0.7
nothing = prediction[0][3] > 0.7 
player_input =True
#get the random choice from computer 
game_options= [ rock, paper, scissors]
computer_input = random.choice(game_options)

if prediction1[0] == True:
    player_input == rock
elif prediction1[1] == True:
    player_input == paper
elif prediction1[2] == True:
    player_input == scissors
else:
    player_input == nothing

cv2.putText(frame,str(player_input), (52,400),cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 5, cv2.LINE_4)
#display the number of rounds
#display the points being collected
#%%
if (player_input == rock1 and computer_input == paper) or (player_input ==paper1 and computer_input==scissors) or (player_input == scissors1 and computer_input ==paper)
    print('player wins')

elif (player_input ==rock1 and computer_input ==rock) or (player_input ==paper1 and computer_input==paper) or (player_input == scissors1 and computer_input ==scissor)
    print('Draw')

elif (player_input ==paper1 and computer_input ==rock) or (player_input ==rock1 and computer_input ==scissors) or (player_input == scissors1 and computer_input ==rock)
    print('computer wins')

else :
    print('try again')
#%%
#take note of the points for both player and the computer
# if player_points>=3 then print player wins
#break
#if computer_points>=3 the print computer wins this round
# break 
