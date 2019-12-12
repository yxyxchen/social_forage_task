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
	expParas['unqHts'] = [40, 25, 22, 2]
	expParas['decsSec'] = 3
	expParas['fbSelfSec'] = 3
	expParas['fbOtherSec'] = 3
	expParas['travelSec'] = 11
	expParas['rwd'] = 2
	expParas['rwdHigh'] = 3
	expParas['rwdLow'] = 1
	expParas['missLoss'] = -2
	expParas['blockSec'] = 30
	expParas['demoBlockSec'] = 30
	hts_ = {
	'rich' : np.array([40, 28, 22, 2, 2, 2, 2]),
	'poor' : np.array([40, 28, 28, 28, 28, 22, 2])
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
		rwdSeq_[condition] = junk[0 : nTrialMax] # here the data selection doesn't include the tail
		# ht sequence
		junk = []
		for i in range(nChunkMax):
			junk.extend(random.sample(list(hts), chunkSize))
		htSeq_[condition] = junk[0 : nTrialMax]
	outputs = {
		"rwdSeq_" : rwdSeq_,
		"htSeq_" : htSeq_
	}
	return outputs

##### create stimuli #####
def getStims(expParas, win):
# create token stimuli
	trashCan = visual.Rect(win = win, width = 0.3, height = (max(expParas['unqHts']) + 2)  * 0.01,
	units = "height", lineWidth = 4, lineColor = [1, 1, 1], fillColor = [1, 1, 1], pos = (0, 0.0))

	recycleSymbol = visual.ImageStim(win, image="recycle.png", units='height', pos=(0.0, 0.0),
		size=0.1, ori=0.0, color = "black")
	
	trashes = {}
	for i in range(max(expParas['unqHts']) + 1):
		keyName = f'trash{i:d}'
		trashes[str(i)] = visual.Rect(win=win, width = 0.295, height = i  * 0.01,
			units = "height", lineWidth = 4, lineColor = [1, 1, 1], fillColor = [0.5, 0.5, 0.5],\
			pos = (0, -(max(expParas['unqHts']) + 2 -i) / 2 * 0.01))

	fbCircle = visual.Circle(win = win, radius=0.1, units = "height", fillColor = "white",\
		lineWidth = 4, lineColor = [0.54509804, -0.78823529, -0.01960784], pos = (0.0, 0.0))

	# create the traveling time bar 
	whiteTimeBar = visual.Rect(win = win, width = expParas['travelSec'] * 0.03, height = 0.03,
	units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [1, 1, 1], pos = (0, -0.35))
	# return outputs
	outputs = {'trashCan' : trashCan, 'recycleSymbol' : recycleSymbol, "trashes" : trashes,\
	'whiteTimeBar' : whiteTimeBar, "fbCircle" : fbCircle}
	return(outputs)


##### create stimuli #####
def getStimsSocial(expParas, win):
	selfCenter = -0.15
# create token stimuli
	trashCan = visual.Rect(win = win, width = 0.3, height = (max(expParas['unqHts']) + 2)  * 0.01,
	units = "height", lineWidth = 4, lineColor = [1, 1, 1], fillColor = [1, 1, 1], pos = (0 , 0.0))

	recycleSymbol = visual.ImageStim(win, image="recycle.png", units='height', pos=(0, 0.0),
		size=0.1, ori=0.0, color = "black")
	
	trashes = {}
	for i in range(max(expParas['unqHts']) + 1):
		keyName = f'trash{i:d}'
		trashes[str(i)] = visual.Rect(win=win, width = 0.295, height = i  * 0.01,
			units = "height", lineWidth = 4, lineColor = [1, 1, 1], fillColor = [0.5, 0.5, 0.5],\
			pos = (0 , -(max(expParas['unqHts']) + 2 -i) / 2 * 0.01))

	fbCircle = visual.Circle(win = win, radius=0.1, units = "height", fillColor = "white",\
			lineWidth = 4, lineColor = [0.54509804, -0.78823529, -0.01960784], pos = (selfCenter, 0.0))

	fbCircleOther = visual.Circle(win = win, radius=0.1, units = "height", fillColor = "white",\
			lineWidth = 4, lineColor = [-0.39607843,  0.14509804, -0.74117647], pos = (-selfCenter, 0.0))

	# create the traveling time bar 
	whiteTimeBar = visual.Rect(win = win, width = expParas['travelSec'] * 0.03, height = 0.03,
	units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [1, 1, 1], pos = (0, -0.35))
	# return outputs
	outputs = {'trashCan' : trashCan, 'recycleSymbol' : recycleSymbol, "trashes" : trashes,\
	'whiteTimeBar' : whiteTimeBar, "fbCircle" : fbCircle, "fbCircleOther" : fbCircleOther}
	return(outputs)


def showTrial(win, expParas, expInfo, expHandler, stims, rwdSeq_, htSeq_, ifDemo):
	if ifDemo:
		blockSec = expParas['demoBlockSec']
	else:
		blockSec = expParas['blockSec']

	# parse stims
	trashCan = stims['trashCan']
	trashes = stims['trashes']
	recycleSymbol = stims['recycleSymbol']
	whiteTimeBar = stims['whiteTimeBar']
	fbCircle = stims['fbCircle']

	# calcualte the number of frames for key events
	nFbFrame = math.ceil((expParas['travelSec'] - expParas['decsSec']) / expInfo['frameDur'])
	fbBeginFIdx = math.ceil(0.5 / expInfo['frameDur'])
	fbEndFIdx = math.ceil((0.5 + expParas['fbSelfSec']) / expInfo['frameDur'])
	nDecsFrame = math.ceil(expParas['decsSec'] / expInfo['frameDur'])
	
	# start the task 
	totalEarnings = 0
	for blockIdx in range(len(expParas['conditions'])):
		condition = expParas['conditions'][blockIdx]
		rwdSeq = rwdSeq_[condition]
		htSeq = htSeq_[condition]
		taskTime = blockSec * blockIdx
		blockTime = 0
		trialIdx = 0

		# change the backgroud color 
		if blockIdx > 0:
			win.color = 'black'

		# create the message
		if blockIdx == 0:
			if ifDemo:
				message = visual.TextStim(win=win, ori=0,
				text= 'Press Any Key to Start the Practice', font=u'Arial', bold = True, units='height',\
				pos=[0, 0], height=0.06, color= 'white', colorSpace='rgb') 
			else:
				message = visual.TextStim(win=win, ori=0,
				text= 'Press Any Key to Start', font=u'Arial', bold = True, units='height',\
				pos=[0, 0], height=0.06, color= 'white', colorSpace='rgb') 				
		else:
			if ifDemo:
				message = visual.TextStim(win=win, ori=0,
				text= 'The First Practice Block Ends \n Press Any Key to Start the Second Block Practice in a New Campus', font=u'Arial', bold = True, units='height',\
				pos=[0, 0], height=0.06, color= 'white', colorSpace='rgb')
			else:
				message = visual.TextStim(win=win, ori=0,
				text= 'The First Block Ends \n Press Any Key to Start the Second Block in a New Campus', font=u'Arial', bold = True, units='height',\
				pos=[0, 0], height=0.06, color= 'white', colorSpace='rgb')				


		# clear all events
		event.clearEvents() 
		# wait for any key to start the game
		responded = False
		while responded == False:
			# detect keys
			keysNow = event.getKeys()
			if len(keysNow) > 0:
				responded = True
			message.draw()
			win.flip()


		# plot the first searching time 
		for frameIdx in range(nFbFrame):
			whiteTimeBar.draw()
			elapsedSec = (frameIdx + 1) * expInfo['frameDur']
			leftSec = expParas['travelSec'] - elapsedSec
			blueTimeBar = visual.Rect(win = win, width = leftSec * 0.03, height = 0.03,\
			units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
			pos = (- elapsedSec * 0.03 / 2, -0.35))
			blueTimeBar.draw()
			win.flip()



		while blockTime < blockSec:
			scheduledHt = htSeq[trialIdx]
			scheduledRwd = rwdSeq[trialIdx]
            
			# wait for the decision 
			responded = False
			frameIdx = 0
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
					responseBlockTime = blockTime + responseRT + expParas['travelSec'] - expParas['decsSec']
				# draw stimuli
				trashCan.draw()
				trashes[str(scheduledHt)].draw()
				recycleSymbol.color = "black"
				recycleSymbol.draw()
				# draw the time bar
				whiteTimeBar.draw()
				leftSec = expParas['decsSec']- (frameIdx + 1) * expInfo['frameDur']
				elapsedSec = expParas['travelSec'] - leftSec
				blueTimeBar = visual.Rect(win = win, width = leftSec * 0.03, height = 0.03,\
					units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
					pos = (- elapsedSec * 0.03 / 2, -0.35))
				blueTimeBar.draw()
				# update the window
				win.flip()
				# update the frame idx
				frameIdx += 1

			# record the response 
			if responded == False:
				response = -1 # -1 for miss
				responseBlockTime = blockTime + expParas['travelSec']
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
						blueTimeBar = visual.Rect(win = win, width = leftSec * 0.03, height = 0.03,\
							units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
							pos = (- elapsedSec * 0.03 / 2, -0.35))
						blueTimeBar.draw()

					else:
						trashCan.draw()
						trashes[str(scheduledHt)].draw()
						recycleSymbol.color = "red"
						recycleSymbol.draw()
						whiteTimeBar.draw()
						leftSec = expParas['decsSec']- (frameIdx + 1) * expInfo['frameDur']
						elapsedSec = expParas['travelSec'] - leftSec
						blueTimeBar = visual.Rect(win = win, width = leftSec * 0.03, height = 0.03,\
							units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
							pos = (- elapsedSec * 0.03 / 2, -0.35))
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


			# update time and total Earnings
			totalEarnings = totalEarnings + trialEarnings
			preTaskTime = taskTime 
			taskTime = taskTime + spentHt + expParas['travelSec']
			blockTime = blockTime + spentHt + expParas['travelSec']	# block time before searching for the next trashcan

			# save the data before searching for the next trashcan
			expHandler.addData('blockIdx',blockIdx + 1) # since blockIdx starts from 0 
			expHandler.addData('trialIdx',trialIdx + 1)
			expHandler.addData('scheduledHt',scheduledHt)
			expHandler.addData('scheduledRwd',scheduledRwd)
			expHandler.addData('spentHt', spentHt)
			expHandler.addData('responseBlockTime', responseBlockTime)
			expHandler.addData('trialEarnings', trialEarnings)
			expHandler.addData('responseRT', responseRT)
			expHandler.addData('blockTime', blockTime)
			expHandler.nextEntry()

			# give the feedback 
			trialEarnText = visual.TextStim(win=win, ori=0,
			text= '' + str(trialEarnings), font=u'Arial', bold = True, units='height',\
			pos=[0, 0], height=0.1,color=[0.54509804, -0.78823529, -0.01960784], colorSpace='rgb') 	
			for frameIdx in range(nFbFrame):			
				whiteTimeBar.draw()
				elapsedSec = (frameIdx + 1) * expInfo['frameDur']
				leftSec = expParas['travelSec'] - elapsedSec
				blueTimeBar = visual.Rect(win = win, width = leftSec * 0.03, height = 0.03,\
				units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
				pos = (- elapsedSec * 0.03 / 2, -0.35))
				blueTimeBar.draw()
				if (frameIdx >= fbBeginFIdx) & (frameIdx < fbEndFIdx):
					fbCircle.draw()
					trialEarnText.draw()
				win.flip()
			# move to the next trial 
			trialIdx = trialIdx + 1
			
	# show the ending massage 
	event.clearEvents() 
	if ifDemo:
		message = visual.TextStim(win=win, ori=0,
		text= 'The Practice Ends \n Press Any Key to Quit', font=u'Arial', bold = True, units='height',\
		pos=[0, 0], height=0.06, color= 'white', colorSpace='rgb') 
	else:
		message = visual.TextStim(win=win, ori=0,
		text= 'The Experiment Ends \n Press Any Key to Quit', font=u'Arial', bold = True, units='height',\
		pos=[0, 0], height=0.06, color= 'white', colorSpace='rgb') 		
	# wait for any key to quit 
	responded = False
	while responded == False:
		# detect keys
		keysNow = event.getKeys()
		if len(keysNow) > 0:
			responded = True
		message.draw()
		win.flip()


	# save the total earnings 
	trialOutput = {'expHandler':expHandler, 'totalEarnings': totalEarnings} 
	win.close()
	return trialOutput

def showTrialSocial(win, expParas, expInfo, thisExp, stims, htSeq, rwdSeq, taskTime):
	selfCenter = -0.15
 # parse stims
	trashCan = stims['trashCan']
	trashes = stims['trashes']
	recycleSymbol = stims['recycleSymbol']
	whiteTimeBar = stims['whiteTimeBar']
	fbCircle = stims['fbCircle']
	fbCircleOther = stims['fbCircleOther']

	# feedback 
	trialEarningsOnGridOther = pd.read_csv("others.csv", header = None)
	trialEarningsOnGridOther = trialEarningsOnGridOther.values
	taskGrid = np.arange(0, expParas['blockSec'] * len(expParas['conditions']) + 1, step = 1)
	
	for i in range(10):
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
			# draw stimuli
			trashCan.draw()
			trashes[str(scheduledHt)].draw()
			recycleSymbol.color = "black"
			recycleSymbol.draw()
			# draw the time bar
			whiteTimeBar.draw()
			leftSec = expParas['decsSec']- (frameIdx + 1) * expInfo['frameDur']
			elapsedSec = expParas['travelSec'] - leftSec
			blueTimeBar = visual.Rect(win = win, width = leftSec * 0.03, height = 0.03,\
				units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
				pos = (- elapsedSec * 0.03 / 2, -0.35))
			blueTimeBar.draw()
			# update the window
			win.flip()
			# update the frame idx
			frameIdx += 1

		# record the response 
		if responded == False:
			response = -1 # -1 for miss
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
					blueTimeBar = visual.Rect(win = win, width = leftSec * 0.03, height = 0.03,\
						units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
						pos = (- elapsedSec * 0.03 / 2 , -0.35))
					blueTimeBar.draw()

				else:
					trashCan.draw()
					trashes[str(scheduledHt)].draw()
					recycleSymbol.color = "red"
					recycleSymbol.draw()
					whiteTimeBar.draw()
					leftSec = expParas['decsSec']- (frameIdx + 1) * expInfo['frameDur']
					elapsedSec = expParas['travelSec'] - leftSec
					blueTimeBar = visual.Rect(win = win, width = leftSec * 0.03, height = 0.03,\
						units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
						pos = (- elapsedSec * 0.03 / 2, -0.35))
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
		nBlankFrame1 = math.ceil(0.5 / expInfo['frameDur'])
		nFbFrameSelf = math.ceil(expParas['fbSelfSec'] / expInfo['frameDur'])
		nBlankFrame2 = math.ceil(0.5 / expInfo['frameDur'])
		nFbFrameOther = math.ceil(expParas['fbOtherSec'] / expInfo['frameDur'])
		nBlankFrame3 = math.ceil(3 / expInfo['frameDur'])
		
		startFrame = 0
		for frameIdx in range(nBlankFrame1):
			whiteTimeBar.draw()
			elapsedSec = (frameIdx + 1) * expInfo['frameDur']
			leftSec = expParas['travelSec'] - elapsedSec
			blueTimeBar = visual.Rect(win = win, width = leftSec * 0.03, height = 0.03,\
			units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
			pos = (- elapsedSec * 0.03 / 2, -0.35))
			blueTimeBar.draw()
			win.flip()

		startFrame = frameIdx
		for frameIdx in range(startFrame, startFrame + nFbFrameSelf):
			fbCircle.draw()
			trialEarnText.draw()
			whiteTimeBar.draw()
			elapsedSec = (frameIdx + 1) * expInfo['frameDur']
			leftSec = expParas['travelSec'] - elapsedSec
			blueTimeBar = visual.Rect(win = win, width = leftSec * 0.03, height = 0.03,\
			units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
			pos = (- elapsedSec * 0.03 / 2, -0.35))
			blueTimeBar.draw()
			win.flip()

		startFrame = frameIdx
		for frameIdx in range(startFrame, startFrame + nBlankFrame2):
			whiteTimeBar.draw()
			elapsedSec = (frameIdx + 1) * expInfo['frameDur']
			leftSec = expParas['travelSec'] - elapsedSec
			blueTimeBar = visual.Rect(win = win, width = leftSec * 0.03, height = 0.03,\
			units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
			pos = (- elapsedSec * 0.03 / 2, -0.35))
			blueTimeBar.draw()
			win.flip()

		startFrame = frameIdx
		for frameIdx in range(startFrame, startFrame + nFbFrameOther):
			fbCircleOther.draw()
			trialEarnTextOther.draw()
			whiteTimeBar.draw()
			elapsedSec = (frameIdx + 1) * expInfo['frameDur']
			leftSec = expParas['travelSec'] - elapsedSec
			blueTimeBar = visual.Rect(win = win, width = leftSec * 0.03, height = 0.03,\
			units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
			pos = (- elapsedSec * 0.03 / 2, -0.35))
			blueTimeBar.draw()
			win.flip()

		startFrame = frameIdx
		for frameIdx in range(startFrame, startFrame + nBlankFrame3):
			whiteTimeBar.draw()
			elapsedSec = (frameIdx + 1) * expInfo['frameDur']
			leftSec = expParas['travelSec'] - elapsedSec
			blueTimeBar = visual.Rect(win = win, width = leftSec * 0.03, height = 0.03,\
			units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
			pos = (- elapsedSec * 0.03 / 2, -0.35))
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
