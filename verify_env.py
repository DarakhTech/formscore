import mediapipe as mp
import numpy as np
import scipy
import torch
import shap
import captum

print(f"mediapipe:  {mp.__version__}")
print(f"numpy:      {np.__version__}")
print(f"scipy:      {scipy.__version__}")
print(f"torch:      {torch.__version__}")
print(f"shap:       {shap.__version__}")
print(f"captum:     {captum.__version__}")
print(f"CUDA avail: {torch.cuda.is_available()}")