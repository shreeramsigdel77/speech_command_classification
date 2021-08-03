import torch
import os



#save checkpoint
def save_ckp(state, checkpoint_dir, epoch):
    """[How to use]
    #save models
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_ckp(checkpoint, checkpoint_dir,epoch)

    Args:
        state ([type]): [dictionary of checkpoint parameters]
        checkpoint_dir ([type]): [checkpoint output path]
        epoch ([type]): [number of epoch]
    """
    f_path = os.path.join(checkpoint_dir , f'checkpoint_{epoch}.pth')
    torch.save(state, f_path)

#load checkpoint
def load_ckp(checkpoint_fpath, model, optimizer):
    """[How to load pytorch model]
    E.g.
    start_epoch=0
    model = autoencoder().cuda()
    criterion = nn.MSELoss()


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=1e-5)
    if os.path.exists(ckp_path):
        model, optimizer, start_epoch = load_ckp(ckp_path, model, optimizer)


    Args:
        checkpoint_fpath ([type]): [description]
        model ([type]): [description]
        optimizer ([type]): [description]

    Returns:
        [type]: [description]
    """
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']



