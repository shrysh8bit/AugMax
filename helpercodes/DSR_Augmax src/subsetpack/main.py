import torch
from carlini_wagner import carlini_wagner_l2

x = torch.clamp(torch.randn(5, 10), 0, 1)
y = torch.randint(0, 9, (5,))
model_fn = lambda x: x

#targeted
l2v, new_x = carlini_wagner_l2(model_fn, x, 10, targeted=False, y=y)
new_pred = model_fn(new_x)
new_pred = torch.argmax(new_pred, 1)

print(y)
print(new_pred)
print(l2v)
for val in l2v:
  print(val.item())
