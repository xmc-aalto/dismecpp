import pytest
import numpy as np
from scipy import sparse
import pydismec

print(pydismec.reg.SquaredNormConfig(strength=5.0))
print(pydismec.reg.HuberConfig(strength=5.0, epsilon=1.0))
print(pydismec.reg.ElasticConfig(strength=5.0, epsilon=0.1, interpolation=0.1))

wc = pydismec.WeightingScheme.Constant(positive=1, negative=1)
print(pydismec.LossType)
print(help(pydismec.TrainingConfig))
tc = pydismec.TrainingConfig(weighting=wc, regularizer=pydismec.reg.SquaredNormConfig(strength=5.0), loss=pydismec.LossType.SquaredHinge)
print(tc)
tc.weighting = wc
assert False
