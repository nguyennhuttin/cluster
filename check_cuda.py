import torch
import torch.nn as nn
import os
print("Pytorch version:", torch.__version__)


N_TASKS = int(os.getenv('SLURM_NTASKS'))
PROC_ID = int(os.getenv('SLURM_PROCID'))


cuda_available = torch.cuda.is_available()

print("Cuda available:", cuda_available)

if not cuda_available:
    print("Cuda not available for proc", PROC_ID, ":(")
    exit(1)


if torch.cuda.device_count() < PROC_ID + 1:
    print("Not enough cudas available for proc", PROC_ID, ":(")
    print("Cuda count:", torch.cuda.device_count(), "N_TASKS:", N_TASKS)


device = torch.device(f"cuda:{PROC_ID}")

class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output
model = Model(500, 200)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)
    print(model)
	

model.to(device)
print(model)

print(torch.tensor([1,2,3]).to(device))
