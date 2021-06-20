from torchvision.utils import make_grid

def show_tensor_images(ax,
                       image_tensor, 
                       num_images=25, 
                       size=(1, 28, 28), 
                       title='real'):
    image_tensor = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images])
    ax.imshow(image_grid.permute(1, 2, 0).squeeze())
    ax.set_title(title)
    return ax