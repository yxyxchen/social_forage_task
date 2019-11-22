from psychopy import core, visual, gui, data, event
from psychopy.tools.filetools import fromFile, toFile
import time
import numpy as np
import random, math
import os
import sys
import scipy.stats as stats
import pandas as pd

##### expParas #####
def getExpParas():
	expParas = {}
	expParas['conditions'] = ['rich', 'poor']
	expParas['unqHts'] = [18, 13, 10, 2]
	expParas['decsSec'] = 2
	expParas['travelSec'] = 4
	expParas['fbSec'] = 2
	expParas['rwd'] = 2
	expParas['rwdHigh'] = 3.5
	expParas['rwdLow'] = 0.5
	expParas['missLoss'] = -2
	expParas['blockSec'] = 600
	hts_ = {
	'rich' : np.array([18, 13, 10, 2, 2, 2, 2]),
	'poor' : np.array([18, 18, 18, 18, 13, 10, 2])
	}
	expParas['hts_'] = hts_
	return expParas

def getSeqs(expParas):
	blockSec = expParas['blockSec']
	iti = expParas['travelSec']
	rwdHigh = expParas['rwdHigh']
	rwdLow = expParas['rwdLow']
	rwd = expParas['rwd']
	unqHts = expParas['unqHts']
	conditions = expParas['conditions']
	hts_ = expParas['hts_']
	# creat new variables
	nCondition = len(conditions)
	chunkSize = len(hts_['rich'])
	nTrialMax = np.ceil(blockSec / iti).astype(int)
	nChunkMax = np.ceil(nTrialMax / chunkSize).astype(int)
	rwds = np.concatenate((np.repeat(rwdLow, chunkSize), np.repeat(rwdHigh, chunkSize)))
	rwdSeq_ = {}
	htSeq_ = {}
	for c in range(nCondition):
		condition = conditions[c]
		hts = hts_[condition]
		# reward sequence 
		junk = []
		for i in range(np.ceil(nChunkMax / 2).astype(int)):
			junk.extend(random.sample(list(rwds), chunkSize * 2)) 
		rwdSeq_[condition] = junk[1 : nTrialMax]
		# ht sequence
		junk = []
		for i in range(nChunkMax):
			junk.extend(random.sample(list(hts), chunkSize))
		htSeq_[condition] = junk[1 : nTrialMax]
	outputs = {
		"rwdSeq_" : rwdSeq_,
		"htSeq_" : htSeq_
	}
	return outputs

##### create stimuli #####
def getStims(expParas, win):
# create token stimuli
	trashCan = visual.Rect(win = win, width = 0.12, height = (max(expParas['unqHts']) + 2)  * 0.015,
	units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [0, 0, 0], pos = (0, -0.0))

	recycleSymbol = visual.ImageStim(win, image="recycle.eps", units='height', pos=(0.0, 0.0),
		size=0.2, ori=0.0)
	
	trashes = {}
	for i in range(max(expParas['unqHts']) + 1):
		keyName = f'trash{i:d}'
		trashes[str(i)] = visual.Rect(win=win, width = 0.12, height = i  * 0.03,
			units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [0.5, 0.5, 0.5], pos = (0, -0.15))

	# create the traveling time bar 
	whiteTimeBar = visual.Rect(win = win, width = expParas['travelSec'] * 0.06, height = 0.03,
	units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [1, 1, 1], pos = (0, -0.35))
	# create the baseLine
	baseLine = visual.Rect(win = win, width = 0.4, height = 0.01,\
	units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [1, 1, 1],pos = (0, -0.1))
	# outputs 
	outputs = {'trashCan' : trashCan, 'recycleSymbol' : recycleSymbol, "trashes" : trashes, 'whiteTimeBar' : whiteTimeBar,\
	'baseLine' : baseLine}
	return(outputs)

def showTrial(win, expParas, expInfo, thisExp, stims, htSeq, rwdSeq, taskTime, isSocial):
    # parse stims
	trashCan = stims['trashCan']
	trashes = stims['trashes']
	recycleSymbol = stims['recycleSymbol']
	whiteTimeBar = stims['whiteTimeBar']
	baseLine = stims['baseLine']

	# feedback 
	if isSocial:
		trialEarningsOnGridOther = pd.read_csv("others.csv", header = None)
		trialEarningsOnGridOther = trialEarningsOnGridOther.values
		taskGrid = np.arange(0, expParas['blockSec'] * len(expParas['conditions']) + 1, step = 1)
	
	for i in range(2):
		scheduledHt = htSeq[i]
		scheduledRwd = rwdSeq[i]

		# wait for the decision 
		responded = False
		frameIdx = 0
		nDescFrame = math.ceil(expParas['decsSec'] / expInfo['frameDur'])

		while (frameIdx < nDescFrame) and (responded == False):
		    # detect keys
		    keysNow = event.getKeys(keyList={'k', 'd'}, modifiers=False, timeStamped=True)
		    if len(keysNow) > 0:
		        responded = True
		        response = 1 if keysNow[0][0] == "k" else 0 # 1 for accept, 0 for reject 
		        responseRT = (frameIdx + 1) * expInfo['frameDur']
		        responseFrameIdx = frameIdx
		        responseClockTime = keysNow[0][1]
		    # draw stimuli
		    trashCan.draw()
		    recycleSymbol.draw()
		    # draw the time bar
		    whiteTimeBar.draw()
		    leftDecsSec = expParas['decsSec']- (frameIdx + 1) * expInfo['frameDur']
		    elapsedTravelSec = expParas['travelSec'] - leftDecsSec
		    blueTimeBar = visual.Rect(win = win, width = leftDecsSec * 0.06, height = 0.03,\
		    	units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
		    	pos = (- elapsedTravelSec * 0.06 / 2, -0.35))
		    blueTimeBar.draw()
		    # update the window
		    win.flip()
		    # update the frame idx
		    frameIdx += 1
		
		# clear all events
		event.clearEvents() 

		# record the response 
		if responded == False:
			response = -1 # -1 for miss
			responseClockTime = np.nan
			responseRT = np.nan

		# update the decision 
		if responded == True:
		    for frameIdx in range(responseFrameIdx+1, nDescFrame):
		        if response == 1:
		        	trashCan.draw()
		        	recycleSymbol.draw()
		        else:
		        	trashCan.draw()
		        	recycleSymbol.draw()
		        win.flip()

		# count down if the option is accepted
		if response == 1:
		    nCountDownFrame = math.ceil(scheduledHt / expInfo['frameDur'])
		    for frameIdx in range(nCountDownFrame):
		        trashCan.draw()
		        countDownTime = scheduledHt - (frameIdx + 1) * expInfo['frameDur'] # time for the next win flip
		        trash[str(math.floor(countDownTime))].draw()
		        win.flip()

		# trialEarnings and spentHt
		if response == 1:
		    trialEarnings = scheduledRwd
		    spentHt = scheduledHt
		elif response == 0:
		    trialEarnings  = 0
		    spentHt = 0
		else:
		    trialEarnings = expParas['missLoss']
		    spentHt = 0


		# update time 
		preTaskTime = taskTime
		taskTime = taskTime + spentHt + expParas['travelSec']

		# trialEarningsOther
		if isSocial:
			trialEarningsOther = sum(trialEarningsOnGridOther[np.logical_and(taskGrid >= preTaskTime,\
				taskGrid < taskTime)])
			

		# create trialEarnBar

		if(trialEarnings >= 0):
			trialEarnText = visual.TextStim(win=win, ori=0,
			text= '' + str(trialEarnings), font=u'Arial', bold = True, units='height',\
			pos=[0,0], height=0.1,color=[-0.16862745, -0.36470588,  0.27843137], colorSpace='rgb') 
		else:
			trialEarnText = visual.TextStim(win=win, ori=0,
			text= '' + str(trialEarnings), font=u'Arial', bold = True, units='height',\
			pos=[0,0.0], height=0.1,color=[-0.16862745, -0.36470588,  0.27843137], colorSpace='rgb') 		
		if isSocial:
			trialEarnText.pos = [-0.15, 0]
			trialEarnTextOther = visual.TextStim(win=win, ori=0,
				text= '' + str(round(trialEarningsOther[0], 1)), font=u'Arial', bold = True, units='height',\
				pos=[0.15,0], height=0.1,color=[0.9921569, 0.7019608, -0.0745098], colorSpace='rgb') 


	# give the feedback 
		nFdFrame = math.ceil(expParas['fbSec'] / expInfo['frameDur'])
		for frameIdx in range(nFdFrame):
		    #baseLine.draw()
		    #trialEarnBar.draw()
		    trialEarnText.draw()
		    if isSocial:
		    	trialEarnTextOther.draw()
		    whiteTimeBar.draw()
		    elapsedTravelSec = (frameIdx + 1) * expInfo['frameDur']
		    leftTravelSec = expParas['travelSec'] - elapsedTravelSec
		    blueTimeBar = visual.Rect(win = win, width = leftTravelSec * 0.06, height = 0.03,\
		    units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
		    pos = (- elapsedTravelSec * 0.06 / 2, -0.35))
		    blueTimeBar.draw()
		    win.flip()

		# data logging
		# thisExp.addData('BlockNumber',blockIdx+1)
		# thisExp.addData('TrialNumber',trialIdx)
		# thisExp.addData('ScheduledDelay',scheduledDelay)
		# thisExp.addData('ReadyOnsetTime',readyOnsetTime)
		# thisExp.addData('TokenOnsetTime',tokenOnsetTime)
		# thisExp.addData('RewardOnsetTime',rewardOnsetTime)
		# thisExp.addData('PriorTrialFeedbackOnsetTime',feedbackOnsetTime)
		#     # the current trial's feedback hasn't yet appeared
		# thisExp.addData('ResponseClockTime',responseClockTime)
		# thisExp.addData('TrialEarnings',trialEarnings)
		# thisExp.addData('TotalEarned',totalEarned)
		# thisExp.nextEntry()

