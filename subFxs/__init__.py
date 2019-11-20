from psychopy import core, visual, gui, data, event
from psychopy.tools.filetools import fromFile, toFile
import time
import numpy as np
import random, math
import os
import sys
import scipy.stats as stats

##### expParas #####
def getExpParas():
	expParas = {}
	expParas['conditions'] = ['rich', 'poor']
	expParas['unqHts'] = [18, 13, 10, 2]
	expParas['decsSec'] = 2
	expParas['travelSec'] = 4
	expParas['fbSec'] = 2
	expParas['rwdMu'] = 2
	expParas['rwdSigma'] = 0.1
	expParas['rwdUpper'] = 2.2
	expParas['rwdLower'] = 1.8
	expParas['missLoss'] = -2
	expParas['bsSec'] = expParas['travelSec'] - expParas['fbSec'] - expParas['decsSec']
	return expParas


##### create stimuli #####
def getStims(expParas, win):
# create token stimuli
	tokens = {}
	tokens['grey'] = visual.Circle(win=win, radius=0.1, edges=64, units='height',
	    lineWidth=2, lineColor=[1,1,1], fillColor=[0.5,0.5,0.5], pos=(0, 0), interpolate=True,
	    name='grey')
	tokens['red'] = visual.Circle(win=win, radius=0.1, edges=64, units='height',
	    lineWidth=2, lineColor=[1,1,1], fillColor=[0.68627451, -0.62352941, -0.69411765], pos=(0, 0), interpolate=False,
	    name='red')
	tokens['green'] = visual.Circle(win=win, radius=0.1, edges=64, units='height',
	    lineWidth=2, lineColor=[1,1,1], fillColor=[-0.79607843,  0.19215686, -0.37254902], pos=(0, 0), interpolate=False,
	    name='green')
	# create countDown stimuli 
	countDowns = {}
	for i in range(max(expParas['unqHts'])):
	    keyName = f'countDown{i:d}'
	    if i > 0:
	        countDowns[str(i)] =visual.TextStim(win=win, ori=0, name= keyName,
	            text= str(i) + "s", font=u'Arial', bold = True, units='height',
	            pos=[0, 0], height=0.12, wrapWidth=None,
	            color=[1,1,1], colorSpace='rgb', opacity=1, depth=0.0)
	    else:
	        countDowns[str(i)] =visual.ImageStim(win=win, ori=0, name= keyName,
	        image = "check.png",
	        pos=[0, 0], size = 0.2)

	# create the traveling time bar 
	whiteTimeBar = visual.Rect(win = win, width = expParas['travelSec'] * 0.06, height = 0.03,
	units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [1, 1, 1],\
	pos = (0, -0.35))
	# create the baseLine
	baseLine = visual.Rect(win = win, width = 0.4, height = 0.01,\
	units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [1, 1, 1],pos = (0, -0.1))
	# outputs 
	outputs = {'tokens' : tokens, 'countDowns' : countDowns, 'whiteTimeBar' : whiteTimeBar,\
	'baseLine' : baseLine}
	return(outputs)

def showTrial(win, expParas, expInfo, thisExp, stims, scheduledHt, scheduledRwd):

    # parse stims
	tokens = stims['tokens']
	countDowns = stims['countDowns']
	whiteTimeBar = stims['whiteTimeBar']
	baseLine = stims['baseLine']

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
	    tokens['grey'].draw()
	    countDowns[str(scheduledHt)].draw()
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
	            tokens['green'].draw()
	            countDowns[str(scheduledHt)].draw()
	        else:
	            tokens['red'].draw()
	            countDowns[str(scheduledHt)].draw()
	        win.flip()

	# count down if the option is accepted
	if response == 1:
	    nCountDownFrame = math.ceil(scheduledHt / expInfo['frameDur'])
	    for frameIdx in range(nCountDownFrame):
	        tokens['green'].draw()
	        countDownTime = scheduledHt - (frameIdx + 1) * expInfo['frameDur'] # time for the next win flip
	        countDowns[str(math.floor(countDownTime))].draw()
	        win.flip()


	# feedback 
	if response == 1:
	    trialEarnings = scheduledRwd
	elif response == 0:
	    trialEarnings  = 0
	else:
	    trialEarnings = expParas['missLoss']
	# create trialEarnBar
	# trialEarnBar = visual.Rect(win = win, width = 0.09, height = trialEarnings * 0.09,\
	# units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16862745, -0.36470588,  0.27843137],\
	# pos = (0, trialEarnings * 0.045 - 0.1))
	# # create trialEarnText
	if(trialEarnings >= 0):
		trialEarnText = visual.TextStim(win=win, ori=0,
		text= 'You Get ' + str(trialEarnings), font=u'Arial', bold = True, units='height',\
		pos=[0,0], height=0.1,color=[-0.16862745, -0.36470588,  0.27843137], colorSpace='rgb') 
	else:
		trialEarnText = visual.TextStim(win=win, ori=0,
		text= 'No Response! ' + str(trialEarnings), font=u'Arial', bold = True, units='height',\
		pos=[0,0], height=0.1,color=[-0.16862745, -0.36470588,  0.27843137], colorSpace='rgb') 		
	# trialEarnings * 0.045 - 0.1

# give the feedback 
	nFdFrame = math.ceil(expParas['fbSec'] / expInfo['frameDur'])
	for frameIdx in range(nFdFrame):
	    #baseLine.draw()
	    #trialEarnBar.draw()
	    trialEarnText.draw()
	    whiteTimeBar.draw()
	    elapsedTravelSec = (frameIdx + 1) * expInfo['frameDur']
	    leftTravelSec = expParas['travelSec'] - elapsedTravelSec
	    blueTimeBar = visual.Rect(win = win, width = leftTravelSec * 0.06, height = 0.03,\
	    units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137],\
	    pos = (- elapsedTravelSec * 0.06 / 2, -0.35))
	    blueTimeBar.draw()
	    win.flip()



	# data logging
	thisExp.addData('BlockNumber',blockIdx+1)
	thisExp.addData('TrialNumber',trialIdx)
	thisExp.addData('ScheduledDelay',scheduledDelay)
	thisExp.addData('ReadyOnsetTime',readyOnsetTime)
	thisExp.addData('TokenOnsetTime',tokenOnsetTime)
	thisExp.addData('RewardOnsetTime',rewardOnsetTime)
	thisExp.addData('PriorTrialFeedbackOnsetTime',feedbackOnsetTime)
	    # the current trial's feedback hasn't yet appeared
	thisExp.addData('ResponseClockTime',responseClockTime)
	thisExp.addData('TrialEarnings',trialEarnings)
	thisExp.addData('TotalEarned',totalEarned)
	thisExp.nextEntry()

