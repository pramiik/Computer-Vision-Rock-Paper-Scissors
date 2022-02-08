#%%
import random

def play():
    game_options= [ 'rock', 'paper', 'scissors']
    computer_input = random.choice(game_options)
    print (f"computer's choice is {computer_input}")
    
    player_input = input()

    print(f"player's choice is {player_input}")


    while True:
        if (player_input == 'rock' and computer_input == 'paper') or (player_input == 'paper' and computer_input=='scissors') or (player_input == 'scissors' and computer_input == 'paper'):
            print('player wins!')
            
        elif (player_input ==  'rock'   and computer_input == 'rock' ) or (player_input =='paper' and computer_input=='paper') or (player_input == 'scissors' and computer_input =='scissor'):
            print('Draw!')

        elif (player_input == 'paper'  and computer_input == 'rock' ) or (player_input == 'rock' and computer_input =='scissors') or (player_input == 'scissors' and computer_input == 'rock' ):
            print('computer wins! ')

        else:
            print('try again')

        
        if ord('q'):
            break


play()
# %%
