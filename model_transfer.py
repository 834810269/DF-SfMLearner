import torch
import models
import argparse

parser = argparse.ArgumentParser(description='model transfer',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input', metavar='DIR', help='path to raw model')
parser.add_argument('--output', metavar='DIR', help='path to transfered model')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
args = parser.parse_args()
model = models.DispResNet().to(device)
weights = torch.load(args.input)
model.load_state_dict(weights['state_dict'])
torch.save(model.state_dict(), args.output,_use_new_zipfile_serialization=False)