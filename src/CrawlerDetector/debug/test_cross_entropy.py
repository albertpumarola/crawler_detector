import torch
from torch.autograd import Variable

criterion_bce = torch.nn.BCELoss()

input = torch.nn.functional.sigmoid(Variable(torch.randn(3), requires_grad=True))
target = Variable(torch.FloatTensor(3).random_(2))

input = Variable(torch.FloatTensor([1,1,0.5]), requires_grad=True)
target = Variable(torch.ones(3))

loss = criterion_bce(input, target)
loss.backward()

print input
print target
print loss


m = torch.nn.Sigmoid()
input = Variable(torch.randn(2, 3, 4, 4))
print(input)
print(m(input))