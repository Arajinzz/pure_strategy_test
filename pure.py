import numpy as np
import pandas as pd
from math import inf

print('CHOOSE EXAMPLE')
print('1. Example 1')
print('2. Example 2')
print('3. Example 3')
print('4. Example 4')
print('5. Prisonners Dilemma')

choice = int(input('Your Choice : '))

if(choice == 1):
    # Strategies For player1 and player 2
    STRATEGIES_P1 = ['T', 'M', 'B']
    STRATEGIES_P2 = ['L', 'C', 'R']

    #gains for Strategies column by column
    GAINS = np.array([1,2,2,2,2,1,2,3,2,1,0,0,0,3,3,2,1,0])
elif(choice == 2):
    STRATEGIES_P1 = ['H', 'M', 'B']
    STRATEGIES_P2 = ['G', 'M', 'D']

    GAINS = np.array([4,3,2,1,3,0,5,1,8,4,9,6,6,2,3,6,2,8])
elif(choice == 3):
    STRATEGIES_P1 = ['A', 'B']
    STRATEGIES_P2 = ['X', 'Y']

    GAINS = np.array([2, 2, 0, 2, -2, 1, -1, 2])
elif(choice == 4):
    STRATEGIES_P1 = ['A', 'B', 'C']
    STRATEGIES_P2 = ['X', 'Y', 'Z']

    GAINS = np.array([3,1,2,0,0,-1,-1,2,4,-3,0,2,0,-1,-2,4,1,1])
else:
    STRATEGIES_P1 = ['Silent', 'Blame']
    STRATEGIES_P2 = ['Silent', 'Blame']

    GAINS = np.array([-1, -1, 0, -5, -5, 0, -3, -3])



# get good shape of game
# constructe the table of game
def constructGame(SP1, SP2, G):
    GAME = {}
    STR_GAINS = []
    
    for i in np.arange(0, G.shape[0], 2):
        STR_GAINS.append(G[i:i+2])
    
    begin = 0
    spacing = len(SP1)
    
    for label in SP2:
        GAME[label] = STR_GAINS[begin:begin+spacing]
        begin += spacing
    
    GAME = pd.DataFrame(GAME, index = SP1)
    
    return GAME


GAME = constructGame(STRATEGIES_P1, STRATEGIES_P2, GAINS)

print('\n\n\nMATRICE DE JEUX\n')
print(GAME)

# 2 strict dominated
# 1 weakly dominated
# 0 not dominated
def rowcolDominated(p1, p2, ind, isStrict):
    dominated = 2
    for i in range(len(p1)):
        
        # stricly dominated
        if(isStrict and p1[i][ind] >= p2[i][ind]):
            return 0
        
        if(not isStrict and p1[i][ind] > p2[i][ind]):
            return 0
        
        # weakly dominated
        if(not isStrict and p1[i][ind] == p2[i][ind]):
            dominated = 1
        
        
    return dominated


# see if column is strictly dominated or weakly dominated
def checkCol(C, cName, key, df, isStrict):
    df = df.copy()
    cols = df.columns.values
    
    for col in cols:
        # check strictly dominated exclusifly
        if rowcolDominated(C, df[col], 1, isStrict) != 0 and isStrict:
            addNode(dominanceSet, key, 'COLUMN ' + cName + ' Strictly Dominated by ' + col + ' ---> Remove COLUMN ' + cName)
            return True
        
        # check weakly dominated
        if rowcolDominated(C, df[col], 1, isStrict) == 2 and not isStrict:
            addNode(dominanceSet, key, 'COLUMN ' + cName + ' Strictly Dominated by ' + col + ' ---> Remove COLUMN ' + cName)
            return True
        
        if rowcolDominated(C, df[col], 1, isStrict) == 1 and not isStrict:
            addNode(dominanceSet, key, 'COLUMN ' + cName + ' Weakly Dominated by ' + col + ' ---> Remove COLUMN ' + cName)
            return True
        
    return False


# see if row is strictly dominated or weakly dominated
def checkRow(R, rName, key, df, isStrict):
    df = df.copy()
    rows = df.index.values
    
    for row in rows:
        # check strictly dominated exclusifly
        if rowcolDominated(R, df.loc[row, :], 0, isStrict) != 0 and isStrict:
            addNode(dominanceSet, key, 'ROW ' + rName + ' Strictly Dominated by ' + row + ' ---> Remove ROW ' + rName)
            return True
        
        # check weakly dominated or strictly
        if rowcolDominated(R, df.loc[row, :], 0, isStrict) == 2 and not isStrict:
            addNode(dominanceSet, key, 'ROW ' + rName + ' Strictly Dominated by ' + row + ' ---> Remove ROW ' + rName)
            return True
        
        if rowcolDominated(R, df.loc[row, :], 0, isStrict) == 1 and not isStrict:
            addNode(dominanceSet, key, 'ROW ' + rName + ' Weakly Dominated by ' + row + ' ---> Remove ROW ' + rName)
            return True
        
    return False



game_cpy = GAME.copy()
dominanceSet = {}

def addNode(D, key, discreption):
    D[key] = discreption

    
# find all dominant strategis
# isStrict mode for finding strictly dominant strategy
# dominanceSet is where we store a Tree of all operations happened and results
# G is the game Table
# step is the operation number
def dominantStrategy(G, step, isStrict):
    
    rows, cols = (G.index.values, G.columns.values)
    values = []
    
    if G.empty:
        return
     
    # get the result dominant strategie
    if(len(rows) == 1 and len(cols) == 1):
        addNode(dominanceSet, step, '('+rows[0]+','+cols[0]+')')
    
    
    # iterate through columns
    for col in cols:
        # copy to not lose the dataframe (game table)
        df = G.copy()
        
        # the column we try to remove
        cCheck = df[col]
        df = df.drop(columns=[col])
        
        # check if we can remove
        if(checkCol(cCheck, col, step, df, isStrict)):
            # if removed recursive call with new sub operation id
            # and new Table the one with the removed column
            dominantStrategy(df, step+'.1', isStrict)
            
            # get new operation id
            step = step.split('.')
            step[-1] = str(int(step[-1]) + 1)
            step = '.'.join(step)
            
    
    # iterate through rows
    # same things as columns
    for row in rows:
        df = G.copy()
        rCheck = df.loc[row, :]
        df = df.drop([row])
        
        if(checkRow(rCheck, row, step, df, isStrict)):
            dominantStrategy(df, step+'.1', isStrict)
            step = step.split('.')
            step[-1] = str(int(step[-1]) + 1)
            step = '.'.join(step)
    
    
    
#dictionary (dominanceSet) is immutable object
#it will change
dominantStrategy(game_cpy, 'operation.1', False)


def reformulate(S, key):
    
    temp = []
    # strategies found
    strats = []
    
    for key in S:
        tkey = key.split('.')
        if(len(tkey) > 2 and int(tkey[-1]) > 1):
            tkey = tkey[:-1]
            howmuch = len(tkey) - 2
            #backtrack
            for i in range(howmuch):
                ttkey = tkey[0:i-howmuch]
                temp.append(S['.'.join(ttkey)])
            temp.append(S['.'.join(tkey)])
            
        temp.append(S[key])
    
    # prints steps and save dominance Strategies
    for s in temp:
        if(s[0] == '('):
            if(s not in strats):
                strats.append(s)
            print(s+'\n')
        else:
            print(s)
    
    return strats

print('\n\nDOMINANTE STRATEGIES ELIMINATION\n\n') 
SS = reformulate(dominanceSet, 'operation.1')
dominanceSet = None


def getBestR(G, rows, cols, isP1, ind):
    playerBest = []
    
    for sc in cols:
        col = None
        
        if isP1:
            col = G[sc]
        else:
            col = G.loc[sc, :]
        
        tmax = -inf
        bestR = []
        
        for sr in rows:
            if(col[sr][ind] == tmax):
                if isP1:
                    bestR.append('({},{})'.format(sr, sc))
                else:
                    bestR.append('({},{})'.format(sc, sr))
            
            if(col[sr][ind] > tmax):
                bestR.clear()
                
                if isP1:
                    bestR.append('({},{})'.format(sr, sc))
                else:
                    bestR.append('({},{})'.format(sc, sr))
                    
                tmax = col[sr][ind]
        
        for i in bestR:
            playerBest.append(i)
        
    return playerBest
    

def nash_equilibrium(G):
    
    rows, cols = (G.index.values, G.columns.values)
    
    player1 = []
    player2 = []
    
    #player 1
    player1 = getBestR(G, rows, cols, True, 0)
    player2 = getBestR(G, cols, rows, False, 1)
            
    equilibriums = []
    
    for c in player1:
        if c in player2:
            equilibriums.append(c)
    
    return equilibriums



game_cpy = GAME.copy()

NASH = nash_equilibrium(game_cpy)

print('\n\nNASH EQUILIBRIUM')
print(NASH)


def pareto_dominated(G):
    
    rows, cols = (G.index.values, G.columns.values)
    
    desc = []
    dominated = []
    
    for c in cols:
        col = G[c]
        for r in rows:
            row = col[r]
            for cc in cols:
                coll = G[cc]
                for rr in rows:
                    roww = coll[rr]
                    if (roww[0] > row[0] and roww[1] >= row[1]) or (roww[0] >= row[0] and roww[1] > row[1]):
                        dominated.append('({}.{})'.format(r, c))
                        desc.append('({}.{})'.format(r, c)+' dominated by '+'({}.{})'.format(rr, cc))
    
    return (desc, list(set(dominated)))


game_cpy = GAME.copy()

desc, dominated_pareto = pareto_dominated(game_cpy)

print('\n\nPARETO DOMINATION')
print(desc)

def pareto_optimum(G, pdominated):
    
    rows, cols = (G.index.values, G.columns.values)
    
    optimums = []
    
    for c in cols:
        for r in rows:
            op = '({}.{})'.format(r, c)
            if(op not in pdominated):
                optimums.append(op)
                
    return optimums


print('\n\nOPTIMUMS PARETO')
pare = pareto_optimum(game_cpy, dominated_pareto)
print(pare)


def security_level(G):
    
    rows, cols = G.values.shape
    
    player1 = []
    player2 = []
    
    for i in range(rows):
        t = []
        for j in range(cols):
            t.append(G.values[i, j][0])
        
        player1.append(t)
    
    
    for i in range(cols):
        t = []
        for j in range(rows):
            t.append(G.values[j, i][1])
        
        player2.append(t)
    
    player1 = [min(i) for i in player1]
    player2 = [min(i) for i in player2]
    
    max1 = max(player1)
    max2 = max(player2)
    
    player1 = [i for i, x in enumerate(player1) if x == max1]
    player2 = [i for i, x in enumerate(player2) if x == max2]
    
    return (G.index.values[player1], G.columns.values[player2])
    

sec = security_level(game_cpy)
print('\n\nSECURITY LEVEL')
print('player1 security ' + sec[0])
print('player2 security ' + sec[1])

