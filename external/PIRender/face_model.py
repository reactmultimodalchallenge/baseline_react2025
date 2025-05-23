import functools
import torch
import torch.nn as nn
from external.PIRender import flow_util
from .base_function import LayerNorm2d, ADAINHourglass, FineEncoder, FineDecoder


class FaceGenerator(nn.Module):
    def __init__(self, ):
        super(FaceGenerator, self).__init__()
        self.mapping_net = MappingNet()
        self.warpping_net = WarpingNet()
        self.editing_net = EditingNet()

    def forward(
        self,
        input_image, 
        driving_source, 
        # stage=None,
        ):
        descriptor = self.mapping_net(driving_source)
        output = self.warpping_net(input_image, descriptor)
        output['fake_image'] = self.editing_net(input_image, output['warp_image'], descriptor)
        return output


class MappingNet(nn.Module):
    def __init__(self, flame_coeff_nc = 58, coeff_nc = 73, descriptor_nc = 256, layer = 3):
        super( MappingNet, self).__init__()

        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)

        self.pre = torch.nn.Conv1d(flame_coeff_nc, coeff_nc, kernel_size=1, padding=0, bias=True)

        self.first = nn.Sequential(
            torch.nn.Conv1d(coeff_nc, descriptor_nc, kernel_size=7, padding=0, bias=True))

        for i in range(layer):
            net = nn.Sequential(nonlinearity,
                torch.nn.Conv1d(descriptor_nc, descriptor_nc, kernel_size=3, padding=0, dilation=3))
            setattr(self, 'encoder' + str(i), net)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc

    def forward(self, input_3dmm):
        out = self.pre(input_3dmm)
        out = self.first(out)
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) + out[:,:,3:-3]
        out = self.pooling(out)
        return out


class WarpingNet(nn.Module):
    def __init__(
        self, 
        image_nc = 3,
        descriptor_nc = 256,
        base_nc = 32,
        max_nc = 256,
        encoder_layer = 5 ,
        decoder_layer = 3,
        use_spect = False
        ):
        super( WarpingNet, self).__init__()

        nonlinearity = nn.LeakyReLU(0.1)
        norm_layer = functools.partial(LayerNorm2d, affine=True) 
        kwargs = {'nonlinearity':nonlinearity, 'use_spect':use_spect}

        self.descriptor_nc = descriptor_nc 
        self.hourglass = ADAINHourglass(image_nc, self.descriptor_nc, base_nc,
                                       max_nc, encoder_layer, decoder_layer, **kwargs)

        self.flow_out = nn.Sequential(norm_layer(self.hourglass.output_nc), 
                                      nonlinearity,
                                      nn.Conv2d(self.hourglass.output_nc, 2, kernel_size=7, stride=1, padding=3))

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input_image, descriptor):
        final_output={}
        output = self.hourglass(input_image, descriptor)
        final_output['flow_field'] = self.flow_out(output)

        deformation = flow_util.convert_flow_to_deformation(final_output['flow_field'])
        final_output['warp_image'] = flow_util.warp_image(input_image, deformation)
        return final_output


class EditingNet(nn.Module):
    def __init__(
        self, 
        image_nc = 3,
        descriptor_nc = 256,
        layer = 3,
        base_nc = 64,
        max_nc = 256,
        num_res_blocks = 2,
        use_spect = False):
        super(EditingNet, self).__init__()

        nonlinearity = nn.LeakyReLU(0.1)
        norm_layer = functools.partial(LayerNorm2d, affine=True) 
        kwargs = {'norm_layer':norm_layer, 'nonlinearity':nonlinearity, 'use_spect':use_spect}
        self.descriptor_nc = descriptor_nc

        # encoder part
        self.encoder = FineEncoder(image_nc*2, base_nc, max_nc, layer, **kwargs)
        self.decoder = FineDecoder(image_nc, self.descriptor_nc, base_nc, max_nc, layer, num_res_blocks, **kwargs)

    def forward(self, input_image, warp_image, descriptor):
        x = torch.cat([input_image, warp_image], 1)
        x = self.encoder(x)
        gen_image = self.decoder(x, descriptor)
        return gen_image


if __name__ == '__main__':
    model = FaceGenerator()
    x = torch.randn(1,3,256,256)
    latent = torch.randn(1, 58, 100)
    output = model(x, latent)
    print(f"output shape: {output['fake_image'].shape}")

    checkpoint = torch.load('/lustre/projects/Research_Project-T127204/xk219/projects/Human-AI/main/'
                            'react-challenge-2025/PIRender/result/face/epoch_01500_iteration_000313500_checkpoint.pt')
    state_dict = checkpoint['net_G_ema']
    print(f"type of state_dict: {type(state_dict)}")

    torch.save(state_dict,
               "/lustre/projects/Research_Project-T127204/xk219/projects/Human-AI/main/react-challenge-2025/"
               "external/PIRender/cur_model_fold.pth")

    # print every parameter name and its shape
    for name, tensor in state_dict.items():
        print(f"{name:<60} {tuple(tensor.shape)}")
    model.load_state_dict(state_dict)
    print(f"model: {model}")