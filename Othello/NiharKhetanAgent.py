'''
Created on October 12, 2014
File Name: nkhetan.py

Usage: % python gameplay.py [-t<timelimit>] [-v] player1 player2
Example: python gameplay.py -t64 -v nkhetan simpleGreedy
Output: Game is Won, lost or Tied

Core:
Evaluation function: To compute the heuristic value at leaf nodes
MiniMax implementation with alpha beta pruning
@author: NiharKhetan

:::  ====>>         STRATEGY         <<====  :::

My Strategy is to plan my next move depending upon both time and current chance which is going on:
    Some key observations on which I designed my strategy:
        - I have defined game in 3 stages:
            Early Stage : Move 1 to 3
            Middle Stages: 4 to 9 and 9 to 14
            Later stages: 15 to 30
        - in early phase of the game searching at lower depths is beneficial as I save time to search at greater depth at later stages of the game
        - in early phase its good to have minimum of my color coins on board as I can reap benefits during middle stage of game
        - in early phase if I can capture middle four squares of the board I keep my chances open for all four diagonals
        - during middle stages if I can limit my opponents moves on board then:
            -I restrict my opponent
            -As well as force him to make wrong moves (no choice)
        - immediately after forcing my opponent in such a situation I must again be greedy and reap benefits and try to get most important positions in the game
        - Most important positions in the game are the CORNERS because they cannot be flipped and can win you edges thus winning the game
        - A balance need to be maintained at all stages between time and depth to which I am searching to prevent from running out of time
        - If time is below 8 then in order to prevent timeout do minimal computations and just return the best move following a evaluation matrix which can 
           determine a good move at current situation
        
        Each of this stage is elaborated more in comments in the code 
'''

import gamePlay
from copy import deepcopy
from gamePlay import newBoard

def nextMove(board, color, time, reversed = False): 
    '''
    @param board: current board state
            color: current color who has to play the turn
            time: time remaining before end of game
            reversed
    @return: next move which current color should make: (x,y)
    This function call minimax with alpha beta pruning which helps to determine the next best move
    '''
    #Generate possible moves and check if moves are possible then only call minimax   
    moves = generatePossibleMoves(board, color)    
    if len(moves) == 0:
        return "pass"
    elif len(moves) == 1:
        return moves[0]
    #Current state represented as MAX node   
    maxNode = makeNode(board, [], 0, color)
    chanceCount = computeChanceCount(board)   
    
    #Initial phase of game : try to capture center four squares keep your coins less so that you can maximize during middle game    
    if time < 6:
        bestMove = followEvaluationFunction(maxNode)
        return bestMove
    else:
        if chanceCount < 4:
            bestmove = minimaxFunction(maxNode, 3, chanceCount)
        #middle stage of the game. I alternate my moves on chances 6 9 and 12 to save time
        elif chanceCount < 13:            
            bestmove = minimaxFunction(maxNode, 5, chanceCount)            
        #handling time for game chance 14 to 19. Whenever time is falling below a threshold I decrease the depth     
        elif chanceCount < 20:
            if time < 20:
                bestmove = minimaxFunction(maxNode, 3, chanceCount)
            else:
                bestmove = minimaxFunction(maxNode, 5,chanceCount)
        #handling time and depth for last stages of the game
        else:
            if time < 5:
                bestmove = minimaxFunction(maxNode, 1, chanceCount)   
            elif time < 18:
                bestmove = minimaxFunction(maxNode, 3, chanceCount)
            elif time < 23:
                bestmove = minimaxFunction(maxNode, 5, chanceCount)         
            elif time < 28:
                bestmove = minimaxFunction(maxNode, 7, chanceCount) 
            else:  
                bestmove = minimaxFunction(maxNode, 5, chanceCount) 
        return bestmove

'''
    MINIMAX STRATEGY
    I used the high level concept of algorithm from Russel and Norvig from Chapter 5
    I have divided it in three parts as a:
    lowerbound is set because minimax drives from root or the MAX node
    MinimaxFunction which always is the driver and it starts assuming itself to be MAX node and calls --> chanceOfMin (Min player or opponent should play)
    upperbound is set as this is MIN node, always chooses lowest values
    chanceOfMin : This tries to minimize or give MAX worst possible move. It calls --> chanceOfMax (Max play should play)
    lowerbound is set as this is MAX node, always chooses highest values
    chanceOfMax : This tries to maximize the gains and tried to choose best move in perspective of MAX. It calls --> chanceIfMin (as after Max again min plays)
    When depth is reached evaluationFunction is called which returns with a heuristic value.
    
    FOR MINIMAX
    
    minimax(s)                                                                   {Implemented as minimaxFunction returns currentBestMove}
    utility(s)  -->   Terminal test                                              {Implemented as evaluationFunction}
    Max a belongs to all Action(s) minimax(result(s,a)) if player(s) - MAX       {Implemented as chanceOfMax}
    Min a belongs to all Action(s) minimax(result(s,a)) if player(s) - MIN       {Implemented as chanceOfMin}
    
    Always minimizes
    checks for chanceOfMin:
        if calculatedScore < currentBestScore:            
            currentBestScore = calculatedScore
    return  currentBestScore
    
    always maximizes
    checks for chanceOfMax:
        if calculatedScore > currentBestScore:            
            currentBestScore = calculatedScore
    return  currentBestScore
    
    alpha beta pruning check comments in code
'''

def minimaxFunction(initialNode, maxDepth, chanceCount):
    '''
    @param initialNode: MAX node
            maxDepth: maximum depth to which tree has to be expanded as DFS
            chanceCount: count of chance which is going on 
    @return: currentBestMove
    This function is the driver function which starts to expand the root nrode or the MAX node
    This calls chanceOfMin <--> chanceOfMax recursively to traverse as DFS and simultaneously back up max and min scored at respective levels
    '''
    #making a move 
    possibleMoves = generatePossibleMoves(initialNode[0], initialNode[3])
    currentBestMove = possibleMoves[0]
    #lower bound at this is MAX node so anything backed up will be greater than lower bound     
    currentBestScore = -99999
    for eachMove in possibleMoves:
        newBoard = deepcopy(initialNode[0])
        gamePlay.doMove(newBoard, initialNode[3], eachMove)
        #making a new node out of new board generated
        newNodeGenerated = makeNode(newBoard, initialNode[0], initialNode[2]+1, invertColor(initialNode[3]))    
        #calls chanceOfMin as next turn is of MIN node or the opponent
        #alphaValue is lower bound on minimax score
        #betaValue is upper bound on minimax score
        calculatedScore = chanceOfMin(newNodeGenerated, maxDepth, -99999, 99999, chanceCount)
        #if calculated score is greater than current best that means a better move is found so choose that
        if calculatedScore > currentBestScore:
            currentBestMove = eachMove
            currentBestScore = calculatedScore
    return currentBestMove

def chanceOfMin(minNode, maxDepth, alphaScore, betaScore, chanceCount):
    '''
    Play as a MIN node or Opponent node. It tried to select worst possible move for max.
    
    alpha beta implementation:
    here beta value is the upper bound on exact minimax score --> it will only decrease
    here alpha value is the lower bound on exact minimax score --> it can only increase
    After score generated by --> chanceOfMin() that is max plays a turn 
    if min function can choose score which is less or equal to alpha
     this means that nodes further can be pruned and chosen score is the new lower bound for alpha
    otherwise beta being an upper bound can be set to new upper bound which is the chosen score 
    @param minNode: MIN node to be expanded
            maxDepth: maxDepth to search before evaluating the base condition
            alphaScore: alpha scores
            betaScore: beta scores
            chanceCount: count of chance which is going on
    @return: currentBestScores 
    '''
    if minNode[2] == maxDepth:
        return evaluationFunction(minNode,chanceCount)
    possibleMoves = generatePossibleMoves(minNode[0], minNode[3])
    #upper bound for min
    currentBestScore = 99999
    for eachMove in possibleMoves:
        newBoard = deepcopy(minNode[0])
        gamePlay.doMove(newBoard, minNode[3], eachMove)
        #calls chanceOfMax as next turn is of MAX node 
        newNodeGenerated = makeNode(newBoard, minNode[0], minNode[2]+1, invertColor(minNode[3]))
        calculatedScore = chanceOfMax(newNodeGenerated, maxDepth, alphaScore, betaScore, chanceCount)
        if calculatedScore < currentBestScore:            
            currentBestScore = calculatedScore
        #alpha is the upper bound for max node to be found so if minnode finds a lower score then subsequent nodes can be pruned
        if currentBestScore <= alphaScore:
            return currentBestScore
        if currentBestScore < betaScore:
            betaScore = currentBestScore
    return currentBestScore

def chanceOfMax(maxNode, maxDepth, alphaScore, betaScore, chanceCount):
    '''
    Play as a MAX node . It tried to select best possible move for max.
    
    alpha beta implementation:
    here alpha value is the lower bound on exact minimax score --> it can only increase
    here beta value is the upper bound on exact minimax score --> it will only decrease
    After score generated by --> chanceOfMin() that is min plays a turn 
    if max function can choose score which is greater that or equal to beta
     this means that nodes further can be pruned and chosen score is the new upper bound for beta
    otherwise alpha being a lower bound can be set to new lower bound which is the chosen score 
    @param maxNode: MAX node to be expanded
        maxDepth: maxDepth to search before evaluating the base condition
        alphaScore: alpha scores
        betaScore: beta scores
        chanceCount: count of chance which is going on
    @return: currentBestScores    
    '''
    if maxNode[2] == maxDepth:
        return evaluationFunction(maxNode, chanceCount)
    possibleMoves = generatePossibleMoves(maxNode[0], maxNode[3])
    #lower bound for max
    currentBestScore = -99999
    for eachMove in possibleMoves:
        newBoard = deepcopy(maxNode[0])
        gamePlay.doMove(newBoard, maxNode[3], eachMove)
        newNodeGenerated = makeNode(newBoard, maxNode[0], maxNode[2]+1, invertColor(maxNode[3]))
        #calls chanceOfMin as next turn is of MIN node or the opponent
        calculatedScore = chanceOfMin(newNodeGenerated, maxDepth, alphaScore, betaScore, chanceCount)
        if calculatedScore > currentBestScore:            
            currentBestScore = calculatedScore
        #beta is the lower bound for min node to be found so if maxnode finds a better score then subsequent nodes can be pruned
        if currentBestScore >= betaScore:
            return currentBestScore
        if currentBestScore > alphaScore:
            alphaScore = currentBestScore
    return currentBestScore


def makeNode(state, parent, depth, color):
    '''
    @param state: current state of the board
            parent: state if the board before current move
            depth: current depth traversed
            color: color of current move
    @return: makes a node as a list
    '''
    return [state,parent,depth,color]

def invertColor(color):
    '''
    @param color: current color passes -> 'B' or 'W'
    @return: color if B then W --> else B 
    '''
    if color == 'B':
        return 'W'
    else:
        return 'B'  

def evaluationFunction(node, chanceCount):
    '''evaluationFunction to determine the strategy to determine the scores at leaf nodes
    These scores help the MAX node at root to make the best possible move
    
     
    @param node: current node for which evaluation Score or utility score has to be returned
            chanceCount: count of chance which is going on
    @return: evalScore
    '''
    currentBoard = node[0];
    parentBoard = node[1];    
    color = node[3];    
    evalScore = 0 
    #initial stage of game   
    if (chanceCount < 3):
        evalScore = numberOfCurrentColorCoins(currentBoard, invertColor(color), True) + fillCenterOfBoard(currentBoard, invertColor(color))
    #be greedy and make best moves on board to make up for first 4 moves
    elif (chanceCount < 9):
        #normalizing the values
        evalScore = (0.95*numberOfCurrentColorCoinsVSOpposite(currentBoard, invertColor(color))) + (0.05*evaluationMatrix(currentBoard, invertColor(color)))
    #middle stage of the game. Limit opponents move
    elif (chanceCount < 14):        
        evalScore = minimizeOpponentsMoves(currentBoard, parentBoard, color)
    #final stage of the game, be greedy and makes move wisely on the basis of evaluation matrix
    else:
        #normalizing the values
        evalScore = (0.7*numberOfCurrentColorCoinsVSOpposite(currentBoard, invertColor(color))) + (0.9*evaluationMatrix(currentBoard, invertColor(color)))
    return evalScore

def fillCenterOfBoard(board, color):
    '''
    This function checks is the move which has happened for Max has made (3,3) (3,4) (4,3) (4,4) positions of max's color or not
    @param board: current state of the game
            color: color of current move
    @return: value which is a high score
    '''
    value = 0
    if (board[3][3] == invertColor(color) and board[3][4] == invertColor(color) and board[4][3] == invertColor(color) and board[4][4] == invertColor(color)):
        value = 10
    return value   

def numberOfCurrentColorCoinsVSOpposite(board, color):
    '''
    This function calculates the effective coins on board that is : Max's coins - Min's coins
    @param board: current state of the game
            color: color of current move
    @return: value which is the net value of Max's coins minus Min's coins
    '''
    value = 0
    for eachRow in board:
        for eachElement in eachRow:
            if eachElement == color:
                value = value + 1
            elif eachElement == invertColor(color):
                value = value - 1
    return value

def evaluationMatrix(board, color):
    '''
    Evaluation matrix a board replicated matrix which has weighted score distribution depending upon favorable moves.
        Corners and edges are given importance
        Bad moves are marked with negative
        
    This function calculates the net score depending on the board state
    
    @param board: current state of the game
            color: color of current move
    @return: evalScore which is the net score
    '''
    
    evalMatrix = [
                 [ 20, -3,  11,  8,  8,  11,  -3,  20],
                 [-3,  -7, -4,   1,  1,  -4,  -7,  -3],
                 [ 11, -4,  2,   2,  2,   2,  -4,  11], 
                 [ 8,   1,  2,   0,  0,   2,   1,   8],
                 [ 8,   1,  2,   0,  0,   2,   1,   8],
                 [ 11, -4,  2,   2,  2,   2,  -4,  11],
                 [-3,  -7, -4,   1,  1,  -4,  -7, -3],
                 [ 20, -3,  11,  8,  8,  11,  -3,  20]                        
                 ]
    value = 0
    for i in range(8):
        for j in range(8):
            if board[i][j] == color:
                value = value + evalMatrix[i][j]
            elif board[i][j] == invertColor(color):
                value = value - evalMatrix[i][j]
    return value

def numberOfCurrentColorCoins(board, color, ifMinimize):
    '''
    Counts the number of coins of given color in the given board
    @param board: currentState of board
            color: color which needs to be counted on board
    @return: count of color on board
    '''
    value = 0;
    for eachRow in board:
        for eachElement in eachRow:
            if eachElement == color:
                value = value + 1;
    if ifMinimize == True:
        return (value*(-1))
    else:
        return value;
    
def computeChanceCount(board):
    '''
    function computes which chance is currently going on in perspective of a player. Range(1,30)
    @param board: current state of board
    @return: chance
    '''
    chance = 1
    #start with -4, as 4 coins already on board
    countOfCoins = -4
    for i in range(8):
        for j in range(8):
            if board[i][j] == 'W' or board[i][j] == 'B':
                countOfCoins = countOfCoins + 1
    return (chance + (countOfCoins/2))         
    
def minimizeOpponentsMoves(currentBoard, parentBoard, color):
    '''
    This returns the moves possible for the opponent. It multiplies it by -1 as we want to minimize them. 
    More favorable is to limit the opponents moves so that MAX can force opponent to do mistakes
    @param currentboard: current state of the board
            parentBoard: state from which current state is derived
            color: color which has to make a move
    for depth 3 : max -> min -> max -> min ('W' turn) assume 'W' has to go for next move so generate moves and minimize them;
    As Minimun('W') <--> MOST favorable for MAX or 'B'        
    '''    
    generatePossibleMovesForNode = generatePossibleMoves(currentBoard, color)
    #minimizing the moves
    number = (len(generatePossibleMovesForNode)*(-1))
    return number

def followEvaluationFunction(node):
    possibleMoves = generatePossibleMoves(node[0], node[3])
    bestMove = possibleMoves[0]
    bestScore = -99999
    for eachMove in possibleMoves:
        newBoard = deepcopy(node[0])
        gamePlay.doMove(newBoard, node[3], eachMove)
        evalScore = evaluationMatrix(newBoard, node[3])
        if evalScore > bestScore:
            bestScore = evalScore
            bestMove = eachMove
    return bestMove

def generatePossibleMoves(board, color):
    '''
    Function generates possible moves of a color for a given state of board and returns them as a list
    @param board: current state of board
            color: turn of color; 'W' or 'B'
    @return: moves generated as a list
    '''
    moves = []
    for i in range(8):
        for j in range(8):
            if gamePlay.valid(board, color, (i,j)):
                moves.append((i,j))
    return moves

    