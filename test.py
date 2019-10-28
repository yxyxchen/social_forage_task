from psychopy import core, visual, gui, data, event
from psychopy.tools.filetools import fromFile, toFile
import time
import numpy, random

# collect participant info
expName = 'Social_Forage'
expInfo = {'participant':'test', 'social_info_condition':'0'}
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False:
    core.quit() 
expInfo['expName'] = expName
expInfo['date'] = time.strftime("%d%m%Y")

# 