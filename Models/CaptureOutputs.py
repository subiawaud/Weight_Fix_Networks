import torch

class CaptureOutputs:
    def __init__(self):
        self.ave =  torch.zeros(1)
        self.std = torch.zeros(1)



    def fake_conv(self, input, weight, bias, ks, padding, stride, output_size):
        out_ch, in_ch, f_h, f_w = weight.size()
        bs, in_ch, h, w = input.size()
        width = int(((w-ks[0] + 2*padding[0])/(stride[0]))+1) # we determine the width and height of the conv
        height = int(((h-ks[1] + 2*padding[1])/(stride[1]))+1)

        #un folding turns this into the toplitz from
        unfolded = torch.functional.F.unfold(input, kernel_size=ks, padding=padding, stride=stride).cpu().detach()
        print('u size ', unfolded.size())
        print('weight size ', weight.data.size())
        unfolded = unfolded.permute(1,0,2) # to become size w*h, c, num
        before = unfolded.size()
        t = unfolded.reshape(in_ch, f_h, f_w, -1).cpu().detach()
        print('t size', t.size())
        t_m = torch.mean(t, dim = -1) # take the mean over all of the
        t_s = torch.std(t, dim = -1)
        print('t_m size ', t_m.size())
        t_m, t_s = t_m.view(-1).unsqueeze(0), t_s.view(-1).unsqueeze(0)
        print('t_m unsqueezed', t_m.size())
#        return t_m, t_s

        f_f = weight.data.view(weight.data.size()[0], -1).cpu()
        print(f_f.size())
        back = t.reshape(before)#.permute(1,0,2)
        back = back.permute(1,0,2)
        print(back.size(), ' This is the back size')
        test = f_f @ back

        print('before vs after')
        print(test.size())
        print(output_size)
        fold = torch.nn.Fold(output_size=output_size, kernel_size=ks, padding = padding, stride = stride)
        folded = fold(test)
        print(folded.size())
        print(folded)
        return t, torch.zeros(1)

#        print('unfold', unfolded.size())
#        filter = weight.data
#        flat_filter = filter.view(filter.size()[0], -1)
#        print('ff', flat_filter.size())
#        result = flat_filter @ unfolded
#        return(result.view(-1, filter.size()[0], width, height))



    def __call__(self, module, module_in, module_out):

        if isinstance(module, torch.nn.Conv2d):
            print(module_out.size()[-2:])
            res_m, res_s = self.fake_conv(module_in[0], module.weight, module.bias, module.kernel_size, module.padding, module.stride, module_out.size()[-2:])
            print('output size', module_out.size())

            print(module_out)
            self.ave = res_m
            self.std = res_s
            raise exeption
        elif isinstance(module, torch.nn.Linear):
            res = torch.sum(module_in[0], dim = 0) # of size input_dim
            self.ave = torch.sum(module_in[0], dim = 0).cpu()
            self.std = torch.std(module_in[0], dim = 0)

    def summarise(self):
        print(self.ave.size())
        self.clear()

    def clear(self):
        self.inputs = []
