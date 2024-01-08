#!/usr/bin/env python
# coding: utf-8

# In[1]:


from copy import deepcopy                        
import random                                  
from math import exp                            
import numpy as np


# In[2]:


def make_board():
    rows , cols = (board_size,board_size)
    board = []                                       

    for i in range(rows):                         
        col = []
        for j in range(cols):                       
            col.append("*")
        board.append(col)
    return board
    


# In[3]:


def placeQueens(board):                          
    i = 0
    while (i < board_size): #true for the size of board
        row = random.randint(0, board_size - 1) #random number for 0 to 7
        if board[row][i] != "Q" :
            board[row][i] = "Q"
            i+=1


# In[4]:


def getQueens(board):                            # This will get the positions of queens palced in the maze
    queen_positions = []                         
    for i in range(board_size):
        for j in range (board_size):
            if board[i][j] == "Q":
                temp = i,j
                queen_positions.append(temp)
    return queen_positions


# In[5]:


def printBoard(board):                          
    for i in range(board_size):
        for j in range(board_size):
            print(board[i][j], end=' ')
        print()


# In[9]:


def objective_function(board):
    
    # getQueen() return the list of position in the form of tuples
    Q = getQueens(board)
    # initialize horizontal and digonal hits with 0
    horizontal_Hits = 0
    diagonal_Hits = 0
    # For each Queen 
    for i in range(len(Q)):
        for j in range(i+1 , len(Q)):
            if i != j:
                if Q[i][0] == Q[j][0]:
                    horizontal_Hits+=1
            if abs(Q[i][0]-Q[j][0]) == abs(Q[i][1]-Q[j][1]):
                diagonal_Hits+=1
    return(horizontal_Hits,diagonal_Hits)
                
           


# In[11]:



def hillClimbing(board):  
pass
   
def main():
global board_size
board_size = int(input("Enter the board size:"))

board=make_board()
placeQueens(board)
printBoard(board)

print(getQueens(board))

print("\n Objective Function", objective_function(board))


if __name__ == "__main__":
main()


# In[ ]:




