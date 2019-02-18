from test.polynomial import Polynomial
from highway import Highway

p = Polynomial(4)
highway = Highway(4,1, has_relu=False)
highway.init_for_projection()
batch_idx,loss = p.train(highway,lr=0.1)

print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
print('==> Actual function:\t' + Polynomial.poly_desc(p.W_target.view(-1), p.b_target))
print('==> Learned projection:\t' + Polynomial.poly_desc(highway.projLinear.weight.view(-1), highway.projLinear.bias))
print('==> Learned gate:\t' + Polynomial.poly_desc(highway.gateLinear.weight.view(-1), highway.gateLinear.bias))