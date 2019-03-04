from torch.nn.init import xavier_normal_


def weights_init(model):
    """Xavier normal weight initialization for the given model.

    Parameters
    ----------
    model : pytorch model for random weight initialization
    Returns
    -------
    pytorch model with xavier normal initialized weights

    """
    if isinstance(model, nn.Conv2d):
        xavier_normal_(model.weight.data)


def classification_accuracy(model, data_iterator):
    """Calculate the classification accuracy.

    Parameters
    ----------
    model : a pytroch model
    data_iterator : pytorch data_iterator

    Returns
    -------
    accuracy : accuracy of classification

    """
    with torch.no_grad():
        total = 0
        length = 0
        for x, y in data_iterator:
            out_put = model(x.to(device))
            out_put = out_put.cpu().detach()
            total += (out_put.argmax(dim=1)==y.argmax(dim=1)).float().sum()
            length += len(y)
        accuracy = total/length

    return accuracy
