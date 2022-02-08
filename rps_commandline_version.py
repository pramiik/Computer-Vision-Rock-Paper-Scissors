#%%
import random

player_input = input()
#get the random choice from computer 
game_options= [ 'rock', 'paper', 'scissors']
computer_input = random.choice(game_options)
print (computer_input)
game =True
n=[]
p=[]
c=[]

for n in range(5):
    while game is True:

        if (player_input == 'rock' and computer_input == 'paper') or (player_input == 'paper' and computer_input=='scissors') or (player_input == 'scissors' and computer_input == 'paper'):
            player_points= p+1 and print(f"player wins! /n computer: {computer_points} /n player: {player_points}")

        elif (player_input ==  'rock'   and computer_input == 'rock' ) or (player_input =='paper' and computer_input=='paper') or (player_input == 'scissors' and computer_input =='scissor'):
            print(f"Draw/n computer: {computer_points} /n player: {player_points}")

        elif (player_input == 'paper'  and computer_input == 'rock' ) or (player_input == 'rock' and computer_input =='scissors') or (player_input == 'scissors' and computer_input == 'rock' ):
            computer_points = c+1 and print("computer wins! /n computer: {computer_points} /n player: {player_points}")

        else:
            print('try again')

        if ord('q'):
            break
# %%
