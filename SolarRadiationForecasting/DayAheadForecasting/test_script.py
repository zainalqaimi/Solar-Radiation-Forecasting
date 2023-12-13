import numpy as np
import os

from exp.exp_model import Exp_Model

Exp = Exp_Model
setting = 'MLPCopula_L192_H96_encl1_decl1_hdim64_eo16_do4_th32_drop0.0_epochs10_bs64_lr0.0001_lossmse'

print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
Exp.test(setting, 1)
