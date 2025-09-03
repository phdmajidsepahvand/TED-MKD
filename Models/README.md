# Model Checkpoints

Model Checkpoints



The trained models are saved in the `models/checkpoints/` directory. The following checkpoints are available:



* `model\_epoch\_30.pth`: Model after 30 epochs of training.
* `model\_epoch\_50.pth`: Model after 50 epochs of training.



To load a model, use the following code:



```python

import torch

from src.models import CNNTeacherModel



model = CNNTeacherModel()

model.load\_state\_dict(torch.load("models/checkpoints/model\_epoch\_50.pth"))

model.eval()



