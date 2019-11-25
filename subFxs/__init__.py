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
	expParas['fbSelfSec'] = 2
	expParas['fbOtherSec'] = 2
	expParas['travelSec'] = 7
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
	trashCan = visual.Rect(win = win, width = 0.3, height = (max(expParas['unqHts']) + 2)  * 0.02,
	units = "height", lineWidth = 4, lineColor = [1, 1, 1], fillColor = [1, 1, 1], pos = (0, 0.0))

	recycleSymbol = visual.ImageStim(win, image="recycle.png", units='height', pos=(0.0, 0.0),
		size=0.1, ori=0.0, color = "black")
	
	trashes = {}
	for i in range(max(expParas['unqHts']) + 1):
		keyName = f'trash{i:d}'
		trashes[str(i)] = visual.Rect(win=win, width = 0.295, height = i  * 0.02,
			units = "height", lineWidth = 4, lineColor = [1, 1, 1], fillColor = [0.5, 0.5, 0.5],\
			pos = (0, -(max(expParas['unqHts']) + 2 -i) / 2 * 0.02))

	# create the traveling time bar 
	whiteTimeBar = visual.Rect(win = win, width = expParas['travelSec'] * 0.06, height = 0.03,
	units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [1, 1, 1], pos = (0, -0.35))
	# return outputs
	outputs = {'trashCan' : trashCan, 'recycleSymbol' : recycleSymbol, "trashes" : trashes,\
	'whiteTimeBar' : whiteTimeBar}
	return(outputs)


##### create stimuli #####
def getStimsSocial(expParas, win):
	selfCenter = -0.3
# create token stimuli
	trashCan = visual.Rect(win = win, width = 0.3, height = (max(expParas['unqHts']) + 2)  * 0.02,
	units = "height", lineWidth = 4, lineColor = [1, 1, 1], fillColor = [1, 1, 1], pos = (selfCenter , 0.0))

	recycleSymbol = visual.ImageStim(win, image="recycle.png", units='height', pos=(selfCenter , 0.0),
		size=0.1, ori=0.0, color = "black")
	
	trashes = {}
	for i in range(max(expParas['unqHts']) + 1):
		keyName = f'trash{i:d}'
		trashes[str(i)] = visual.Rect(win=win, width = 0.295, height = i  * 0.02,
			units = "height", lineWidth = 4, lineColor = [1, 1, 1], fillColor = [0.5, 0.5, 0.5],\
			pos = (selfCenter , -(max(expParas['unqHts']) + 2 -i) / 2 * 0.02))

	# create the traveling time bar 
	whiteTimeBar = visual.Rect(win = win, width = expParas['travelSec'] * 0.06, height = 0.03,
	units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [1, 1, 1], pos = (selfCenter, -0.35))
	# return outputs
	outputs = {'trashCan' : trashCan, 'recycleSymbol' : recycleSymbol, "trashes" : trashes,\
	'whiteTimeBar' : whiteTimeBar}
	return(outputs)


def showTrial(win, expParas, expInfo, thisExp, stims, htSeq, rwdSeq, taskTime):
 # parse stims
	trashCan = stims['trashCan']
	trashes = stims['trashes']
	recycleSymbol = stims['recycleSymbol']
	whiteTimeBar = stims['whiteTimeBar']
	
	for i in range(2):
		scheduledHt = htSeq[i]
		scheduledRwd = rwdSeq[i]

		# wait for the decision 
		responded = False
		frameIdx = 0
		nDecsFrame = math.ceil(expParas['decsSec'] / expInfo['frameDur'])

		# clear all events
		event.clearEvents() 

		while (frameIdx < nDecsFrame) and (responded == False):
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
			trashes[str(scheduledHt)].draw()
			recycleSymbol.color = "black"
			recycleSymbol.draw()
			# draw the time bar
			whiteTimeBar.draw()
			leftSec = expParas['decsSec']- (frameIdx + 1) * expInfo['frameDur']
			elapsedSec = expParas['travelSec'] - leftSec
			blueTimeBar = visual.Rect(win = win, width = leftSec * 0.06, height = 0.03,\
				units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
				pos = (- elapsedSec * 0.06 / 2, -0.35))
			blueTimeBar.draw()
			# update the window
			win.flip()
			# update the frame idx
			frameIdx += 1

		# record the response 
		if responded == False:
			response = -1 # -1 for miss
			responseClockTime = np.nan
			responseRT = np.nan

		# update the decision 
		if responded == True:
			for frameIdx in range(responseFrameIdx+1, nDecsFrame):
				if response == 1:
					trashCan.draw()
					trashes[str(scheduledHt)].draw()
					recycleSymbol.color = "blue"
					recycleSymbol.draw()
					whiteTimeBar.draw()
					leftSec = expParas['decsSec']- (frameIdx + 1) * expInfo['frameDur']
					elapsedSec = expParas['travelSec'] - leftSec
					blueTimeBar = visual.Rect(win = win, width = leftSec * 0.06, height = 0.03,\
						units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
						pos = (- elapsedSec * 0.06 / 2, -0.35))
					blueTimeBar.draw()

				else:
					trashCan.draw()
					trashes[str(scheduledHt)].draw()
					recycleSymbol.color = "red"
					recycleSymbol.draw()
					whiteTimeBar.draw()
					leftSec = expParas['decsSec']- (frameIdx + 1) * expInfo['frameDur']
					elapsedSec = expParas['travelSec'] - leftSec
					blueTimeBar = visual.Rect(win = win, width = leftSec * 0.06, height = 0.03,\
						units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
						pos = (- elapsedSec * 0.06 / 2, -0.35))
					blueTimeBar.draw()
				win.flip()

		# count down if the option is accepted
		if response == 1:
			nCountDownFrame = math.ceil(scheduledHt / expInfo['frameDur'])
			for frameIdx in range(nCountDownFrame):
				trashCan.draw()
				countDownTime = scheduledHt - (frameIdx + 1) * expInfo['frameDur'] # time for the next win flip
				trashes[str(math.floor(countDownTime))].draw()
				recycleSymbol.color = "blue"
				recycleSymbol.draw()
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
			

		# create trialEarnBar
		trialEarnText = visual.TextStim(win=win, ori=0,
		text= '' + str(trialEarnings), font=u'Arial', bold = True, units='height',\
		pos=[0, 0], height=0.1,color=[0.54509804, -0.78823529, -0.01960784], colorSpace='rgb') 	


	# give the feedback 
		nFbFrameSelf = math.ceil(expParas['fbSelfSec'] / expInfo['frameDur'])
		nBlankFrame = math.ceil((expParas['travelSec'] - expParas['fbSelfSec'] - expParas['decsSec'])/ expInfo['frameDur'])
		for frameIdx in range(nFbFrameSelf):
			trashCan.draw()
			trialEarnText.draw()
			whiteTimeBar.draw()
			elapsedSec = (frameIdx + 1) * expInfo['frameDur']
			leftSec = expParas['travelSec'] - elapsedSec
			blueTimeBar = visual.Rect(win = win, width = leftSec * 0.06, height = 0.03,\
			units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
			pos = (- elapsedSec * 0.06 / 2, -0.35))
			blueTimeBar.draw()
			win.flip()

		for frameIdx in range(nFbFrameSelf, nFbFrameSelf + nBlankFrame):
			trashCan.draw()
			whiteTimeBar.draw()
			elapsedSec = (frameIdx + 1) * expInfo['frameDur']
			leftSec = expParas['travelSec'] - elapsedSec
			blueTimeBar = visual.Rect(win = win, width = leftSec * 0.06, height = 0.03,\
			units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
			pos = (- elapsedSec * 0.06 / 2, -0.35))
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


def showTrialSocial(win, expParas, expInfo, thisExp, stims, htSeq, rwdSeq, taskTime):
	selfCenter = -0.3
 # parse stims
	trashCan = stims['trashCan']
	trashes = stims['trashes']
	recycleSymbol = stims['recycleSymbol']
	recycleMan = stims['recycleMan']
	whiteTimeBar = stims['whiteTimeBar']


	# feedback 
	trialEarningsOnGridOther = pd.read_csv("others.csv", header = None)
	trialEarningsOnGridOther = trialEarningsOnGridOther.values
	taskGrid = np.arange(0, expParas['blockSec'] * len(expParas['conditions']) + 1, step = 1)
	
	for i in range(2):
		scheduledHt = htSeq[i]
		scheduledRwd = rwdSeq[i]

		# wait for the decision 
		responded = False
		frameIdx = 0
		nDecsFrame = math.ceil(expParas['decsSec'] / expInfo['frameDur'])

		# clear all events
		event.clearEvents() 

		while (frameIdx < nDecsFrame) and (responded == False):
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
			trashes[str(scheduledHt)].draw()
			recycleSymbol.color = "black"
			recycleSymbol.draw()
			# draw the time bar
			whiteTimeBar.draw()
			leftSec = expParas['decsSec']- (frameIdx + 1) * expInfo['frameDur']
			elapsedSec = expParas['travelSec'] - leftSec
			blueTimeBar = visual.Rect(win = win, width = leftSec * 0.06, height = 0.03,\
				units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
				pos = (- elapsedSec * 0.06 / 2 + selfCenter, -0.35))
			blueTimeBar.draw()
			# update the window
			win.flip()
			# update the frame idx
			frameIdx += 1

		# record the response 
		if responded == False:
			response = -1 # -1 for miss
			responseClockTime = np.nan
			responseRT = np.nan

		# update the decision 
		if responded == True:
			for frameIdx in range(responseFrameIdx+1, nDecsFrame):
				if response == 1:
					trashCan.draw()
					trashes[str(scheduledHt)].draw()
					recycleSymbol.color = "blue"
					recycleSymbol.draw()
					whiteTimeBar.draw()
					leftSec = expParas['decsSec']- (frameIdx + 1) * expInfo['frameDur']
					elapsedSec = expParas['travelSec'] - leftSec
					blueTimeBar = visual.Rect(win = win, width = leftSec * 0.06, height = 0.03,\
						units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
						pos = (- elapsedSec * 0.06 / 2 + selfCenter, -0.35))
					blueTimeBar.draw()

				else:
					trashCan.draw()
					trashes[str(scheduledHt)].draw()
					recycleSymbol.color = "red"
					recycleSymbol.draw()
					whiteTimeBar.draw()
					leftSec = expParas['decsSec']- (frameIdx + 1) * expInfo['frameDur']
					elapsedSec = expParas['travelSec'] - leftSec
					blueTimeBar = visual.Rect(win = win, width = leftSec * 0.06, height = 0.03,\
						units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
						pos = (- elapsedSec * 0.06 / 2 + selfCenter, -0.35))
					blueTimeBar.draw()
				win.flip()

		# count down if the option is accepted
		if response == 1:
			nCountDownFrame = math.ceil(scheduledHt / expInfo['frameDur'])
			for frameIdx in range(nCountDownFrame):
				trashCan.draw()
				countDownTime = scheduledHt - (frameIdx + 1) * expInfo['frameDur'] # time for the next win flip
				trashes[str(math.floor(countDownTime))].draw()
				recycleSymbol.color = "blue"
				recycleSymbol.draw()
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
		trialEarningsOther = sum(trialEarningsOnGridOther[np.logical_and(taskGrid >= preTaskTime,\
			taskGrid < taskTime)])
			

		# create trialEarnBar
		trialEarnText = visual.TextStim(win=win, ori=0,
		text= '' + str(trialEarnings), font=u'Arial', bold = True, units='height',\
		pos=[selfCenter, 0], height=0.1,color=[0.54509804, -0.78823529, -0.01960784], colorSpace='rgb') 	
		trialEarnTextOther = visual.TextStim(win=win, ori=0,\
			text= '' + str(round(trialEarningsOther[0], 1)), font=u'Arial', bold = True, units='height',\
			pos=[-selfCenter,0], height=0.1,color=[-0.39607843,  0.14509804, -0.74117647], colorSpace='rgb') 


	# give the feedback 
		nFbFrameSelf = math.ceil(expParas['fbSelfSec'] / expInfo['frameDur'])
		nBlankFrame = math.ceil(1 / expInfo['frameDur'])
		nFbFrameOther = math.ceil(expParas['fbOtherSec'] / expInfo['frameDur'])
		for frameIdx in range(nFbFrameSelf):
			trashCan.draw()
			trialEarnText.draw()
			whiteTimeBar.draw()
			elapsedSec = (frameIdx + 1) * expInfo['frameDur']
			leftSec = expParas['travelSec'] - elapsedSec
			blueTimeBar = visual.Rect(win = win, width = leftSec * 0.06, height = 0.03,\
			units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
			pos = (- elapsedSec * 0.06 / 2 + selfCenter, -0.35))
			blueTimeBar.draw()
			win.flip()

		for frameIdx in range(nFbFrameSelf, nFbFrameSelf + nBlankFrame):
			trashCan.draw()
			whiteTimeBar.draw()
			elapsedSec = (frameIdx + 1) * expInfo['frameDur']
			leftSec = expParas['travelSec'] - elapsedSec
			blueTimeBar = visual.Rect(win = win, width = leftSec * 0.06, height = 0.03,\
			units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
			pos = (- elapsedSec * 0.06 / 2 + selfCenter, -0.35))
			blueTimeBar.draw()
			win.flip()

		for frameIdx in range(nFbFrameSelf + nBlankFrame, nFbFrameOther + nFbFrameSelf + nBlankFrame):
			trashCan.draw()
			trialEarnTextOther.draw()
			whiteTimeBar.draw()
			elapsedSec = (frameIdx + 1) * expInfo['frameDur']
			leftSec = expParas['travelSec'] - elapsedSec
			blueTimeBar = visual.Rect(win = win, width = leftSec * 0.06, height = 0.03,\
			units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
			pos = (- elapsedSec * 0.06 / 2 + selfCenter, -0.35))
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
