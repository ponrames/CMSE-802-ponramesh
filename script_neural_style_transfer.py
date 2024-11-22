import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import argparse
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image



def load_image_from_url(url, max_size=400):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    
    max_dim = max(img.size)
    scale_factor = max_size / max_dim
    new_size = tuple([int(x * scale_factor) for x in img.size])
    img = img.resize(new_size, Image.LANCZOS)
    
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    return img_tensor

def resize_image_to_target(content_img, style_img):
    style_img = transforms.Resize(content_img.shape[2:])(style_img)
    return style_img

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean
        self.std = std
    
    def forward(self, img):
        return (img - self.mean[None, :, None, None]) / self.std[None, :, None, None]

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = target_feature.detach()
        self.loss = None

    def forward(self, x):
        gram_x = self.gram_matrix(x)
        gram_target = self.gram_matrix(self.target)
        self.loss = nn.functional.mse_loss(gram_x, gram_target)
        return x

    def gram_matrix(self, x):
        _, c, h, w = x.size()
        x = x.view(c, h * w)
        gram = torch.mm(x, x.t())
        return gram / (c * h * w)

class ContentLoss(nn.Module):
    def __init__(self, target_feature):
        super(ContentLoss, self).__init__()
        self.target = target_feature.detach()
        self.loss = None

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img, content_layers_default, style_layers_default):
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"

        model.add_module(name, layer)

        if name in style_layers_default:
            target_feature = model(style_img)
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

        if name in content_layers_default:
            target_feature = model(content_img)
            content_loss = ContentLoss(target_feature)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

    return model, style_losses, content_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Style Transfer")
    parser.add_argument("--content_url", type=str, required=True, help="URL of the content image")
    parser.add_argument("--style_url", type=str, required=True, help="URL of the style image")
    parser.add_argument("--content_layers_default", type=str, nargs="+", default=['conv_4'], help="List of content layers")
    parser.add_argument("--style_layers_default", type=str, nargs="+", default=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'], help="List of style layers")
    parser.add_argument("--num_steps", type=int, default=500, help="Number of optimization steps")
    parser.add_argument("--style_weight", type=float, default=1000000, help="Weight for style loss")
    parser.add_argument("--content_weight", type=float, default=1, help="Weight for content loss")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = load_image_from_url(args.content_url).to(device)
    style_img = load_image_from_url(args.style_url).to(device)
    style_img = resize_image_to_target(content_img, style_img)

    def image_loader(image):
        return image.clone().detach().requires_grad_(True).to(device)

    content_img = image_loader(content_img)
    style_img = image_loader(style_img)

    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img, args.content_layers_default, args.style_layers_default)

    optimizer = optim.LBFGS([content_img])

    run = [0]

    with tqdm(total=args.num_steps, desc="Style Transfer Progress") as pbar:
        while run[0] <= args.num_steps:
            def closure():
                content_img.data.clamp_(0, 1)
                optimizer.zero_grad()
                model(content_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss

                for cl in content_losses:
                    content_score += cl.loss

                style_score *= args.style_weight
                content_score *= args.content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                pbar.update(1)

                return style_score + content_score

            optimizer.step(closure)

    final_img = content_img.detach().cpu().squeeze(0).permute(1, 2, 0)
    output_filename = "final_style_transferred_image.png"

    # Convert the tensor to a PIL Image and save it
    final_img_pil = to_pil_image(final_img.permute(2, 0, 1))  # Adjust dimensions for to_pil_image
    final_img_pil.save(output_filename)
    print(f"Final image saved as {output_filename}")