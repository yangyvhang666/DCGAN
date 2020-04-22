import torch
from DCModel2 import Generator
import torchvision.utils as vutils
img_list = []
PATH="gen_img/"
ngpu = 1
nz = 100
img_num = 50
model = Generator(ngpu)
model.load_state_dict(torch.load('G_models/netG_9.pth',map_location='cpu'))
model.eval()
print("Lodeing_Success")

for i in range(img_num):
    fixed_noise = torch.randn(64, nz, 1, 1)
    output_image = model(fixed_noise).detach()
#print(output_image.shape)
    img_list.append(vutils.make_grid(output_image,padding=2,normalize=True))
    vutils.save_image(img_list[i],"%s/gen_imgs%d.png"%(PATH,i))
    print("Saving_%d"%(i))

